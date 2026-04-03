# LSTM Stage 2 - LSTM+Attention 精细选股模型

## 📋 概述

这是 ASSTS 量化交易系统的 **Stage 2**（精细选股阶段）。

**输入**: Stage 1 筛选出的候选股票（T日MACD死叉候选池）  
**输出**: 股票涨跌分类概率，输出 Top-K 最具上涨潜力的候选股  
**目标**: 在 MACD 死叉候选池中，进一步筛选出最可能在 **2s3h** 策略下获利的股票

---

## 🗂️ 目录结构

```
lstm_stage/
├── lstm_attention_model.py   # 模型定义、训练、评估、预测
├── lstm_data_prepare.py      # 数据准备脚本（从JSON事件文件生成HDF5）
├── lstm_data_prepare_fast.py # 快速数据准备（轻量版）
├── train_monitor.sh          # 训练监控脚本（后台运行）
├── checkpoints/              # 模型检查点
│   └── best_model.pth
├── data/                     # HDF5训练数据（需运行data_prepare生成）
│   ├── train_data.h5
│   ├── test_data.h5
│   └── data_summary.json
└── logs/                    # 训练日志和预测结果
    ├── train.log
    ├── training_history.json
    └── top_100_predictions.json
```

---

## 🔧 环境依赖

```bash
# 核心依赖
pip install torch numpy pandas h5py scikit-learn

# GPU支持（推荐）
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

**注意**: 训练需要 GPU，强烈建议使用带 NVIDIA GPU 的机器。若无GPU，将自动切换到 CPU（速度较慢）。

---

## 🚀 一键启动（完整流程）

```bash
cd /home/zzy/project/ASSTS/ASSTS-stock_class/lstm_stage

# ========== 步骤1：准备数据 ==========
python lstm_data_prepare.py

# ========== 步骤2：训练模型 ==========
python lstm_attention_model.py --mode train --task regression

# ========== 步骤3：评估模型 ==========
python lstm_attention_model.py --mode eval --checkpoint checkpoints/best_model.pth

# ========== 步骤4：预测TopK候选 ==========
python lstm_attention_model.py --mode predict --checkpoint checkpoints/best_model.pth --top-k 100
```

---

## 📖 详细说明

### 步骤1：数据准备

```bash
# 全量处理（生成 train_data.h5 + test_data.h5）
python lstm_data_prepare.py

# 仅验证数据格式，不生成HDF5
python lstm_data_prepare.py --verify

# 仅处理特定类型
python lstm_data_prepare.py --type 1   # 仅 Type-1
python lstm_data_prepare.py --type 2  # 仅 Type-2
```

**数据来源**:  
- 候选事件JSON文件：`/tmp/strategy1/tushare_output_all-*-1~9-new/`  
- 原始股票日线：`/home/zzy/project/ASSTS/stock_dir/tushare_stock_1107-all/`

**输出**:
- `data/train_data.h5` - 训练集（70%）
- `data/test_data.h5` - 测试集（30%，按时间划分）
- `data/data_summary.json` - 数据统计摘要

**9维输入特征**:
| 索引 | 特征名 | 说明 |
|------|--------|------|
| 0 | open | 开盘价 |
| 1 | high | 最高价 |
| 2 | low | 最低价 |
| 3 | close | 收盘价 |
| 4 | volume | 成交量 |
| 5 | diff | MACD diff |
| 6 | dea | MACD dea |
| 7 | market_return | 大盘收益率（上证指数） |
| 8 | type_id | Type类型 (1/2/3) |

---

### 步骤2：模型训练

```bash
# 回归任务（预测具体d_profit值）
python lstm_attention_model.py --mode train --task regression

# 分类任务（预测涨跌概率）
python lstm_attention_model.py --mode train --task classification
```

**训练参数**:
- SEQ_LEN = 60（时间序列长度，T-60 ~ T-1）
- HIDDEN_DIM = 128 → 64
- BATCH_SIZE = 128
- LEARNING_RATE = 1e-3
- MAX_EPOCHS = 30
- PATIENCE = 10（早停）

**模型架构**:
```
Input(60, 9)
  → LSTM(128, return_sequences=True) + MultiHeadAttention
  → LSTM(64, return_sequences=False) + MultiHeadAttention
  → Dense(64, ReLU) + Dropout(0.3)
  → Dense(32, ReLU)
  → Dense(1)  [regression]
  或
  → Dense(2, softmax)  [classification]
```

**训练输出**:
- `checkpoints/best_model.pth` - 最佳模型
- `logs/training_history.json` - 训练历史
- `logs/train.log` - 训练日志

---

### 步骤3：后台监控训练

```bash
# 启动训练并后台监控
nohup python lstm_attention_model.py --mode train --task regression > train.log 2>&1 &
echo $! > train.pid

# 启动监控脚本（每10分钟汇报一次）
bash train_monitor.sh
```

---

### 步骤4：评估模型

```bash
# 评估测试集
python lstm_attention_model.py --mode eval --checkpoint checkpoints/best_model.pth
```

**输出指标**:
| 任务 | 指标 |
|------|------|
| regression | MAE（平均绝对误差）、Corr（预测相关性） |
| classification | AUC、Accuracy |

---

### 步骤5：预测 TopK 候选

```bash
# 输出预测得分最高的100只股票
python lstm_attention_model.py --mode predict --checkpoint checkpoints/best_model.pth --top-k 100
```

**输出**:
- `logs/top_100_predictions.json` - TopK预测结果
- 包含：股票代码、T日日期、预测d_profit、真实d_profit、Type类型

---

## 🔢 常用命令速查

| 需求 | 命令 |
|------|------|
| 全新训练 | `python lstm_attention_model.py --mode train --task regression` |
| 恢复训练 | `python lstm_attention_model.py --mode train --task regression`（自动加载best_model） |
| 仅评估 | `python lstm_attention_model.py --mode eval --checkpoint checkpoints/best_model.pth` |
| 输出Top100 | `python lstm_attention_model.py --mode predict --checkpoint checkpoints/best_model.pth --top-k 100` |
| 切换GPU | 代码自动检测，无需手动设置 |
| 修改序列长度 | 编辑 `lstm_attention_model.py` 中的 `SEQ_LEN = 60` |

---

## ⚠️ 常见问题

**Q: 报 `ModuleNotFoundError: No module named 'torch'`**  
A: 运行 `pip install torch numpy pandas h5py scikit-learn`

**Q: 报 `HDF5 file not found`**  
A: 先运行 `python lstm_data_prepare.py` 生成数据

**Q: 训练很慢**  
A: 确保使用GPU：`pip install torch --extra-index-url https://download.pytorch.org/whl/cu118`

**Q: 报 `stock CSV not found`**  
A: 检查 `STOCK_DATA_DIR` 路径是否正确，参考 `lstm_data_prepare.py` 中的路径配置

**Q: 如何只训练一只股票测试**  
A: 修改 `lstm_data_prepare.py` 中的 `MAX_SAMPLES_PER_TYPE` 参数

---

## 📊 数据流向

```
原始股票日线数据 (CSV)
    ↓
Stage 1 MACD筛选 (JSON事件文件)
    ↓
lstm_data_prepare.py → HDF5 (60天序列 + 9维特征)
    ↓
lstm_attention_model.py → 训练 → best_model.pth
    ↓
TopK预测 → 输出候选股票列表 → Stage 3 PPO择时
```

---

## 📝 脚本版本说明

| 文件 | 说明 |
|------|------|
| `lstm_data_prepare.py` | 标准版，功能完整 |
| `lstm_data_prepare_fast.py` | 快速版，省略部分验证步骤 |

---

**最后更新**: 2026-04-02
