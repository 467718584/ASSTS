#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 LSTM+Attention 模型

功能:
  - 训练: 从HDF5数据学习预测d_profit
  - 评估: 输出AUC/MAE等指标
  - 预测: 对候选股票排序，输出Top候选

模型结构:
  Input(60, 8)
    → LSTM(128, return_sequences=True) + MultiHeadAttention
    → LSTM(64, return_sequences=False) + MultiHeadAttention
    → Dense(64, ReLU) + Dropout(0.3)
    → Dense(32, ReLU)
    → Dense(1)  [回归版]
    或
    → Dense(2, softmax)  [二分类版]

使用方法:
  python lstm_attention_model.py --mode train --task regression
  python lstm_attention_model.py --mode eval  --checkpoint checkpoints/best_model.pth
  python lstm_attention_model.py --mode predict --checkpoint checkpoints/best_model.pth --top-k 100
"""

import os
import json
import argparse
import time
import random
from datetime import datetime

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================================
# 配置
# ============================================================

BASE_DIR = "/home/zzy/project/ASSTS/ASSTS-stock_class"
LSTM_STAGE_DIR = os.path.join(BASE_DIR, "lstm_stage")
DATA_DIR = os.path.join(LSTM_STAGE_DIR, "data")
CHECKPOINT_DIR = os.path.join(LSTM_STAGE_DIR, "checkpoints")
LOG_DIR = os.path.join(LSTM_STAGE_DIR, "logs")

# 训练超参数
SEQ_LEN = 60
FEATURE_DIM = 9  # 9维特征 (8原始 + 1 type_id)
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
DENSE_DIM = 32
DROPOUT = 0.3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 10
NUM_WORKERS = 0

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {DEVICE}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# 数据集
# ============================================================

class HL5Dataset(Dataset):
    """从HDF5文件加载数据"""
    def __init__(self, h5_path, split="train"):
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        with h5py.File(h5_path, "r") as f:
            # 支持两种格式: h5['train']['features'] 或 h5['features']
            if split in f:
                grp = f[split]
            else:
                grp = f
            self.features = grp["features"][:]
            self.labels = grp["labels"][:]
            self.meta = [m.decode("utf-8") if isinstance(m, bytes) else m for m in grp["meta"][:]]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()   # (60, 9)
        y = torch.tensor(self.labels[idx]).float()          # scalar
        return x, y


class HL5DatasetTopK(Dataset):
    """用于TopK预测的数据集（只返回features和meta）"""
    def __init__(self, h5_path, split="test"):
        super().__init__()
        with h5py.File(h5_path, "r") as f:
            if split in f:
                grp = f[split]
            else:
                grp = f
            self.features = grp["features"][:]
            self.meta = [json.loads(m.decode("utf-8") if isinstance(m, bytes) else m) for m in grp["meta"][:]]
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()
        return x, self.meta[idx]


def collate_fn_topk(batch):
    """自定义collate函数，保证meta不被错误batch"""
    xs, metas = zip(*batch)
    x_batch = torch.stack(xs, dim=0)  # (batch, 60, 8)
    return x_batch, list(metas)


# ============================================================
# 模型：LSTM + MultiHead Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """简单的单头注意力（等同于加权和注意力）"""
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
    
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # 残差连接
        return output + x


class LSTMAttention(nn.Module):
    """
    LSTM + Attention 模型
    输入: (batch, seq_len, feature_dim)
    输出: 回归(d_profit值) 或 分类(正/负收益概率)
    """
    def __init__(self, feature_dim=FEATURE_DIM, hidden1=HIDDEN_DIM_1, hidden2=HIDDEN_DIM_2,
                 dense_dim=DENSE_DIM, dropout=DROPOUT, task="regression"):
        super().__init__()
        self.task = task
        
        # LSTM层1 + Attention
        self.lstm1 = nn.LSTM(feature_dim, hidden1, batch_first=True, bidirectional=False)
        self.attn1 = MultiHeadAttention(hidden1)
        self.layer_norm1 = nn.LayerNorm(hidden1)
        
        # LSTM层2 + Attention
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True, bidirectional=False)
        self.attn2 = MultiHeadAttention(hidden2)
        self.layer_norm2 = nn.LayerNorm(hidden2)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden2, dense_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_dim, dense_dim // 2)
        
        # 输出层
        if task == "regression":
            self.out = nn.Linear(dense_dim // 2, 1)
        elif task == "classification":
            self.out = nn.Linear(dense_dim // 2, 2)  # 2分类
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        
        # LSTM1 + Attention
        lstm_out1, _ = self.lstm1(x)                        # (batch, seq, hidden1)
        attn_out1 = self.attn1(lstm_out1)
        attn_out1 = self.layer_norm1(attn_out1)
        
        # LSTM2 + Attention
        lstm_out2, _ = self.lstm2(attn_out1)                 # (batch, seq, hidden2)
        attn_out2 = self.attn2(lstm_out2)
        attn_out2 = self.layer_norm2(attn_out2)
        
        # 取最后一个时间步
        last_out = attn_out2[:, -1, :]                        # (batch, hidden2)
        
        # 全连接
        out = F.relu(self.fc1(last_out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        
        # 输出
        if self.task == "regression":
            output = self.out(out)                          # (batch, 1)
        else:
            output = F.softmax(self.out(out), dim=-1)        # (batch, 2)
        
        return output


# ============================================================
# Focal Loss（处理样本不平衡）
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss for 二分类，处理97% vs 3%的样本不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # pred: (batch, 2) 概率
        # target: (batch,) 0或1
        probs = pred[:, 1]  # 正类概率
        ce = F.binary_cross_entropy(probs, target, reduction="none")
        p_t = probs * target + (1 - probs) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * ce
        return loss.mean()


# ============================================================
# 训练
# ============================================================

def train_one_epoch(model, dataloader, optimizer, criterion, task, epoch=1, max_epochs=1):
    model.train()
    total_loss = 0
    total_samples = 0
    batch_count = 0
    t_start = time.time()
    
    for x, y in dataloader:
        batch_count += 1
        # 每50个batch打印一次进度
        if batch_count % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / batch_count * (len(dataloader) - batch_count)
            print(f"  Epoch {epoch}/{max_epochs} | Batch {batch_count}/{len(dataloader)} | ETA: {eta:.0f}s", flush=True)
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(x)
        
        if task == "regression":
            loss = criterion(output.squeeze(), y)
        else:
            loss = criterion(output, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(y)
        total_samples += len(y)
    
    return total_loss / total_samples


def evaluate(model, dataloader, task):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            
            if task == "regression":
                preds = output.squeeze().cpu().numpy()
                loss = criterion(output.squeeze(), y).item()
            else:
                preds = output[:, 1].cpu().numpy()  # 正类概率
                loss = criterion(output, y.long()).item()
            
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
            total_loss += loss * len(y)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / len(all_labels)
    
    # 计算指标
    if task == "regression":
        mae = np.mean(np.abs(all_preds - all_labels))
        # 相关性
        corr = np.corrcoef(all_preds, all_labels)[0, 1] if len(all_labels) > 1 else 0
        metrics = {"mae": mae, "corr": corr, "loss": avg_loss}
    else:
        # AUC
        from sklearn.metrics import roc_auc_score, accuracy_score
        binary_labels = (all_labels > 0).astype(int)
        auc = roc_auc_score(binary_labels, all_preds) if len(set(binary_labels)) > 1 else 0.5
        acc = accuracy_score(binary_labels, (all_preds > 0.5).astype(int))
        metrics = {"auc": auc, "accuracy": acc, "loss": avg_loss}
    
    return metrics, all_preds, all_labels


def train(args):
    """完整训练流程"""
    train_path = os.path.join(DATA_DIR, "train_data.h5")
    if not os.path.exists(train_path):
        print(f"[ERROR] 训练数据不存在: {train_path}")
        print("请先运行: python lstm_data_prepare.py")
        return
    
    task = args.task
    print(f"[INFO] 训练任务: {task}")
    print(f"[INFO] 加载训练数据: {train_path}")
    
    train_dataset = HL5Dataset(train_path, split="train")
    test_path = os.path.join(DATA_DIR, "test_data.h5")
    val_dataset = HL5Dataset(test_path, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"[INFO] 训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")
    
    # 模型
    model = LSTMAttention(task=task).to(DEVICE)
    print(f"[INFO] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和调度器
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # 训练循环
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_metric": []}
    
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        print(f"[INFO] Epoch {epoch}/{MAX_EPOCHS} 开始...", flush=True)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, task, epoch, MAX_EPOCHS)
        val_metrics, _, _ = evaluate(model, val_loader, task)
        val_loss = val_metrics["loss"]
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - t0
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        metric_name = "val_auc" if task == "classification" else "val_mae"
        history["val_metric"].append(val_metrics.get(metric_name.split("_")[1], 0))
        
        # 打印
        if task == "regression":
            print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val MAE: {val_metrics['mae']:.6f} | "
                  f"Val Corr: {val_metrics['corr']:.4f} | "
                  f"{epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"{epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "task": task,
            }, best_path)
            print(f"  ✅ 保存最佳模型: {best_path}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= PATIENCE:
            print(f"\n[INFO] 早停触发 (patience={PATIENCE})")
            break
    
    # 保存训练历史（转换numpy类型）
    history_path = os.path.join(LOG_DIR, "training_history.json")
    history_serializable = {
        k: [float(x) for x in v] for k, v in history.items()
    }
    with open(history_path, "w") as f:
        json.dump(history_serializable, f)
    print(f"[INFO] 训练历史已保存: {history_path}")
    
    print(f"\n✅ 训练完成！最佳验证损失: {best_val_loss:.6f}")


# ============================================================
# 评估
# ============================================================

def eval_model(args):
    """评估模型"""
    if not args.checkpoint:
        print("[ERROR] 请指定 --checkpoint 参数")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    task = checkpoint.get("task", "regression")
    print(f"[INFO] 加载模型: {args.checkpoint}")
    print(f"[INFO] 任务类型: {task}")
    
    model = LSTMAttention(task=task).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 加载测试集
    test_path = os.path.join(DATA_DIR, "test_data.h5")
    if not os.path.exists(test_path):
        test_path = os.path.join(DATA_DIR, "train_data.h5")
    
    dataset = HL5Dataset(test_path, split="test")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"[INFO] 测试集: {len(dataset)} 条")
    
    metrics, preds, labels = evaluate(model, loader, task)
    
    print("\n========== 评估结果 ==========")
    if task == "regression":
        print(f"  MAE (平均绝对误差): {metrics['mae']:.6f}")
        print(f"  Corr (预测相关性):  {metrics['corr']:.4f}")
        print(f"  Loss:               {metrics['loss']:.6f}")
    else:
        print(f"  AUC:                {metrics['auc']:.4f}")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Loss:               {metrics['loss']:.6f}")
    
    # 额外分析：按d_profit分位数看预测效果
    if task == "regression":
        labels_arr = np.array(labels)
        preds_arr = np.array(preds)
        
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        print("\n--- 按真实d_profit分位数分析 ---")
        for i in range(len(quantiles) - 1):
            low = np.quantile(labels_arr, quantiles[i])
            high = np.quantile(labels_arr, quantiles[i+1])
            mask = (labels_arr >= low) & (labels_arr < high)
            if mask.sum() > 0:
                avg_pred = np.mean(preds_arr[mask])
                avg_true = np.mean(labels_arr[mask])
                print(f"  [{quantiles[i]*100:.0f}%~{quantiles[i+1]*100:.0f}%): "
                      f"真实均值={avg_true:.4f}, 预测均值={avg_pred:.4f}, n={mask.sum()}")
    
    # Top-K分析：如果预测值最高的样本，真实收益是否也高？
    if task == "regression":
        top_k = min(1000, len(preds))
        top_indices = np.argsort(preds)[-top_k:]
        top_labels = labels[top_indices]
        print(f"\n--- Top-{top_k} 预测样本分析 ---")
        print(f"  Top预测样本真实d_profit均值: {np.mean(top_labels):.4f}")
        print(f"  Top预测样本正收益占比: {np.mean(top_labels > 0)*100:.2f}%")
        print(f"  全量均值: {np.mean(labels):.4f}, 正收益占比: {np.mean(labels > 0)*100:.2f}%")
    
    print("================================\n")


# ============================================================
# 预测 & TopK排序
# ============================================================

def predict(args):
    """对候选集预测并输出TopK"""
    if not args.checkpoint:
        print("[ERROR] 请指定 --checkpoint 参数")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    task = checkpoint.get("task", "regression")
    print(f"[INFO] 加载模型: {args.checkpoint}")
    
    model = LSTMAttention(task=task).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 加载数据
    data_path = os.path.join(DATA_DIR, "train_data.h5")
    
    dataset = HL5DatasetTopK(data_path, split="test")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=NUM_WORKERS, collate_fn=collate_fn_topk)
    
    print(f"[INFO] 待预测样本: {len(dataset)} 条")
    
    all_preds = []
    all_meta = []
    
    with torch.no_grad():
        for x, meta in loader:
            x = x.to(DEVICE)
            output = model(x)
            if task == "regression":
                preds = output.squeeze().cpu().numpy()
            else:
                preds = output[:, 1].cpu().numpy()
            
            if preds.ndim == 0:
                preds = np.array([preds.item()])
            
            all_preds.extend(preds.tolist())
            all_meta.extend(meta)
    
    # 排序
    sorted_indices = np.argsort(all_preds)[::-1]  # 降序
    
    top_k = args.top_k or min(100, len(sorted_indices))
    top_indices = sorted_indices[:top_k]
    
    print(f"\n========== Top {top_k} 候选 ==========")
    print(f"{'排名':<6}{'股票代码':<10}{'T日日期':<12}{'预测d_profit':<14}{'真实d_profit':<14}{'Type'}")
    print("-" * 70)
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        meta = all_meta[idx]
        pred = all_preds[idx]
        true = meta.get("d_profit", "N/A")
        print(f"{rank:<6}{meta['code']:<10}{meta['t_date']:<12}"
              f"{pred:.6f}     {true:.6f}     {meta['type']}")
        results.append({
            "rank": rank, "code": meta["code"], "t_date": meta["t_date"],
            "pred_d_profit": pred, "true_d_profit": true, "type": meta["type"],
            "d_x": meta.get("d_x", 0)
        })
    
    # 保存结果
    output_path = os.path.join(LOG_DIR, f"top_{top_k}_predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")
    
    # 简单统计
    true_vals = [r["true_d_profit"] for r in results if isinstance(r["true_d_profit"], float)]
    print(f"\nTop-{top_k} 真实d_profit统计:")
    print(f"  均值: {np.mean(true_vals):.4f}")
    print(f"  正收益占比: {np.mean(np.array(true_vals)>0)*100:.2f}%")
    print(f"  最大值: {np.max(true_vals):.4f}, 最小值: {np.min(true_vals):.4f}")
    print("================================\n")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM+Attention Stage2 模型")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "predict"],
                        default="train", help="运行模式")
    parser.add_argument("--task", type=str, choices=["regression", "classification"],
                        default="regression", help="训练任务类型")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型检查点路径（eval/predict模式必填）")
    parser.add_argument("--top-k", type=int, default=None,
                        help="预测模式输出TopK（默认100）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_model(args)
    elif args.mode == "predict":
        predict(args)
