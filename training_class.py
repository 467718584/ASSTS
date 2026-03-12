import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn as nn
import torch
import numpy as np
from Class_score.model.CNN_class import TimeSeriesScorer1DCNN_class
from Class_score.model.LSTM_Attention import TimeSeriesScorerLSTM_Attn
# from Class_score.model.Transformer import TimeSeriesTransformer # 如果你用Transformer
from data_preload import create_data_loaders
import torch.nn.functional as F

# === 新增 Focal Loss 类 ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs 是 logits, targets 是标签
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-bce_loss) # pt 是预测正确的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TimeSeriesTrainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001, device='cpu', pos_weight=1.0, loss_type='BCE',
                 save_type='F1'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_type = save_type

        # === 核心修改 1: 使用二分类 Loss 并加权 ===
        # pos_weight=4.0 意味着模型把"漏掉一个好票"的惩罚看作"做错一个坏票"的4倍
        # 这对于处理 1:4 的数据不平衡非常有效
        weight_tensor = torch.tensor([pos_weight]).to(device)
        if loss_type == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        elif loss_type == 'FOCAL':
            self.criterion = FocalLoss(gamma=2, pos_weight=weight_tensor)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # 监控指标改为 val_loss 或 val_f1
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)

        self.train_losses = []
        self.val_metrics = []  # 记录综合指标
        self.best_val_f1 = 0  # 改用 F1 或 AUC 作为早停依据
        self.best_val_auc = 0

    def calculate_metrics(self, y_true, y_pred_logits):
        """计算二分类指标"""
        # 将 Logits 转为概率和 0/1 标签
        probs = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_true, probs)
        except:
            auc = 0.5

        return acc, prec, rec, f1, auc

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        all_true = []
        all_logits = []

        for batch_X, batch_y in self.train_loader:
            self.optimizer.zero_grad()

            # 简单的数据清洗，防止 NaN
            batch_X = torch.nan_to_num(batch_X, nan=0.0)

            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            logits = self.model(batch_X)
            loss = self.criterion(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()

            # 收集结果用于计算 epoch 指标
            all_true.extend(batch_y.cpu().numpy())
            all_logits.extend(logits.detach().cpu().numpy())

        # 计算训练集指标
        metrics = self.calculate_metrics(all_true, all_logits)
        return epoch_loss / len(self.train_loader), metrics

    def validate(self):
        self.model.eval()
        val_loss = 0
        all_true = []
        all_logits = []

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = torch.nan_to_num(batch_X, nan=0.0)
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                val_loss += loss.item()

                all_true.extend(batch_y.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        metrics = self.calculate_metrics(all_true, all_logits)
        return val_loss / len(self.val_loader), metrics

    def train(self, epochs=100, patience=20):
        early_stop_counter = 0

        print(f"{'Epoch':<5} | {'Train Loss':<10} | {'Val Loss':<10} "
              f"| {'Train F1':<8} | {'Train AUC':<8} | {'Train Acc':<8} " 
              f"| {'Val F1':<8} | {'Val AUC':<8} | {'Val Acc':<8}")
        print("-" * 65)

        for epoch in range(epochs):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            val_acc, val_prec, val_rec, val_f1, val_auc = val_metrics
            train_acc, train_prec, train_rec, train_f1, train_auc = train_metrics

            save_msg = ""

            if self.save_type == 'F1':
                # 学习率调度：监控 F1 分数
                self.scheduler.step(val_f1)

                # === 核心修改 2: 早停机制改为监控 F1 Score ===
                # 因为数据不平衡，Accuracy 可能有欺骗性，F1 或 AUC 更靠谱
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    early_stop_counter = 0
                    save_msg = " [Saved]"
                else:
                    early_stop_counter += 1
                    save_msg = ""

                if early_stop_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}. Best Val F1: {self.best_val_f1:.4f}")
                    break

            elif self.save_type == 'AUC':
                # 学习率调度：监控 AUC 分数
                self.scheduler.step(val_auc)

                # === 核心修改 2: 早停机制改为监控 F1 Score ===
                # 因为数据不平衡，Accuracy 可能有欺骗性，F1 或 AUC 更靠谱
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    early_stop_counter = 0
                    save_msg = " [Saved]"
                else:
                    early_stop_counter += 1
                    save_msg = ""

                if early_stop_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}. Best Val AUC: {self.best_val_auc:.4f}")
                    break

            print(
                f"{epoch:<5} | {train_loss:.6f}   | {val_loss:.6f}   "
                f"| {train_f1:.4f}   | {train_auc:.4f}   | {train_acc:.4f}    "
                f"| {val_f1:.4f}   | {val_auc:.4f}   | {val_acc:.4f}{save_msg}")

        print("Loading best model for testing...")
        self.model.load_state_dict(torch.load('best_model.pth'))


def main_training_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 创建数据加载器
    # 确保你的 create_data_loaders 返回的 y 已经是 0/1 标签
    batch_size = 256  # 二分类可以适当加大 Batch Size
    dataset_name = f'dataset/preprocess/0d-m-data-120d-preprocess-14f-2s3h-class-0005_003.pkl'  # 你的数据文件名
    loss_type = 'BCE'  # BCE/FOCAL
    data_length = 30
    save_type = 'AUC'
    print(f"Loading dataset: {dataset_name}")
    print(f"Loading loss: {loss_type}")
    print(f"Time length: {data_length}")
    train_loader, val_loader, test_loader, X_test, y_test = create_data_loaders(dataset_name, batch_size,
                                                                                feature_length=data_length)

    # ==========================================
    # === 新增：动态计算 pos_weight ===
    # ==========================================
    print("正在计算动态权重...")

    # 获取训练集的所有标签
    # create_data_loaders 返回的 train_loader 是 DataLoader 对象
    # 我们需要访问其底层的 dataset.tensors[1] (即 y_train)
    y_train_tensor = train_loader.dataset.tensors[1]  # y_train shape: (N, 1)

    # 转为 numpy 进行统计
    labels = y_train_tensor.numpy()

    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)
    total = len(labels)

    # 防止除以 0 的保护措施
    if num_pos == 0:
        print("警告：训练集中没有正样本！请检查阈值设置。")
        pos_weight_val = 1.0
    else:
        # 核心公式
        pos_weight_val = num_neg / num_pos

    print(f"数据统计: 正样本={num_pos} ({num_pos / total:.2%}), 负样本={num_neg} ({num_neg / total:.2%})")
    print(f"自动计算的 pos_weight: {pos_weight_val:.4f}")

    # ==========================================

    # 2. 初始化模型
    # ... (自动获取维度的代码) ...
    sample_X, _ = next(iter(train_loader))
    time_steps = sample_X.shape[1]
    input_channels = sample_X.shape[2]
    print(f"Detected Input: Steps={time_steps}, Channels={input_channels}")
    # model = TimeSeriesScorer1DCNN_class(input_channels=input_channels, time_steps=time_steps).to(device)
    model = TimeSeriesScorerLSTM_Attn(input_channels=input_channels, time_steps=time_steps).to(device)


    # 3. 训练模型 (pos_weight 根据你的数据比例调整，推荐 4.0)

    trainer = TimeSeriesTrainer(model, train_loader, val_loader, lr=0.001, device=device, pos_weight=pos_weight_val,
                                loss_type=loss_type, save_type=save_type)

    trainer.train(epochs=100, patience=50)

    # 4. 测试集详细评估
    model.eval()
    all_true = []
    all_logits = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = torch.nan_to_num(batch_X, nan=0.0).to(device)
            logits = model(batch_X)
            all_true.extend(batch_y.numpy())
            all_logits.extend(logits.cpu().numpy())

    # 最终测试报告
    acc, prec, rec, f1, auc = trainer.calculate_metrics(all_true, all_logits)

    print(f"\n===== Final Test Results =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (查准率: 选出的股票有多少是真的好票)")
    print(f"Recall:    {rec:.4f} (查全率: 所有好票里抓住了多少)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    return model, trainer


if __name__ == '__main__':
    main_training_pipeline()