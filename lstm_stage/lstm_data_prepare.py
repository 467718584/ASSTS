#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 LSTM+Attention 数据准备脚本

功能:
  - 加载 Type-1/2/3 JSON 事件文件
  - 从原始股票CSV中提取 T-60~T-1日 的8维特征序列
  - 拼接大盘指数涨跌幅（用上证指数代替）
  - 滑动窗口归一化，避免未来数据泄露
  - 输出 train/test split 的 HDF5 文件

输出文件:
  - train_data.h5   : 训练集
  - test_data.h5    : 测试集（按时间划分，后30%为测试）
  - data_summary.json: 数据集统计摘要

使用方法:
  python lstm_data_prepare.py                    # 全量处理
  python lstm_data_prepare.py --verify          # 仅验证数据，不生成HDF5
  python lstm_data_prepare.py --type 1           # 仅处理Type-1
"""

import os
import json
import glob
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import h5py

warnings.filterwarnings("ignore")

# ============================================================
# 配置路径
# ============================================================
BASE_DIR = "/home/zzy/project/ASSTS/ASSTS-stock_class"
STOCK_DATA_DIR = "/home/zzy/project/ASSTS/stock_dir/tushare_stock_1107-all"
STRATEGY1_DIR = "/tmp/strategy1"
OUTPUT_DIR = os.path.join(BASE_DIR, "lstm_stage", "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "lstm_stage", "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "lstm_stage", "logs")

# 特征配置
SEQ_LEN = 60        # 时间序列长度（T-60 ~ T-1）
FEATURE_DIM = 9     # 9维特征 (8原始 + 1 type_id)
FEATURE_NAMES = ["open", "high", "low", "close", "volume", "diff", "dea", "market_return", "type_id"]

# Type-1/2/3 JSON 目录
TYPE_DIRS = {
    "Type-1": os.path.join(STRATEGY1_DIR, "tushare_output_all-0-1~9-new"),
    "Type-2": os.path.join(STRATEGY1_DIR, "tushare_output_all-1-1~9-new"),
    "Type-3": os.path.join(STRATEGY1_DIR, "tushare_output_all-2-1~9-new"),
}

# 标签字段
LABEL_FIELD = "d_profit"

# 大盘指数代码（用上证指数 000001.SH 或 999999.SH 代替）
INDEX_CODE = "000001"  # 使用平安银行等大盘股作为市场代理，或自行接入指数数据


# ============================================================
# 工具函数
# ============================================================

def parse_date(date_val):
    """解析日期字段，支持float(20100915.0)和str('2010-09-15')两种格式"""
    if isinstance(date_val, (int, float)):
        s = str(int(date_val))
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    elif isinstance(date_val, str):
        return date_val[:10]
    return None


def load_stock_csv(code):
    """加载单只股票的CSV数据，返回DataFrame"""
    # 尝试多种文件命名方式
    for pattern in [
        os.path.join(STOCK_DATA_DIR, f"{code}.csv"),
        os.path.join(STOCK_DATA_DIR, f"{code.lstrip('0')}.csv"),
    ]:
        if os.path.exists(pattern):
            df = pd.read_csv(pattern)
            if "日期" in df.columns:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                    "DIFF": "diff", "DEA": "dea"
                })
            if "date" not in df.columns and "日期" not in df.columns:
                # 尝试第一列作为日期
                df.columns = ["date"] + list(df.columns[1:])
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                return df
    return None


def get_market_return(df, date, window=1):
    """计算date日之前window天的市场收益率（用收盘价变化率代替）"""
    idx = df[df["date"] <= date].index
    if len(idx) < window + 1:
        return 0.0
    price_t = df.loc[idx[-1], "close"]
    price_tn = df.loc[idx[-1 - window], "close"]
    if price_tn == 0 or np.isnan(price_tn):
        return 0.0
    return (price_t - price_tn) / price_tn


def extract_features(df, end_date, seq_len=60):
    """
    从df中提取end_date之前seq_len天的特征序列。
    返回 shape=(seq_len, 8) 的numpy数组（type_id由调用方填充为第9维）
    
    特征: [open, high, low, close, volume, diff, dea, market_return]
    """
    # 取end_date之前的所有数据
    sub_df = df[df["date"] < end_date].tail(seq_len).copy()
    
    if len(sub_df) < seq_len:
        return None  # 数据不足
    
    features = []
    
    for _, row in sub_df.iterrows():
        # 基础OHLCV
        o = row.get("open", np.nan)
        h = row.get("high", np.nan)
        l = row.get("low", np.nan)
        c = row.get("close", np.nan)
        v = row.get("volume", np.nan)
        diff = row.get("diff", 0.0)
        dea = row.get("dea", 0.0)
        
        # 大盘收益率（用当日收盘/前一日收盘近似）
        market_ret = get_market_return(df, row["date"], window=1)
        
        if any(np.isnan(x) for x in [o, h, l, c, v]):
            return None
        
        features.append([o, h, l, c, v, diff, dea, market_ret])
    
    arr = np.array(features, dtype=np.float32)
    
    # 归一化（滑动窗口归一化：每个样本独立用自身均值std）
    # 注意：这里不用全局统计量，避免未来数据泄露
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0  # 防止除零
    arr = (arr - mean) / std
    
    return arr


def process_type_events(type_name, json_dir, limit=None):
    """
    处理单个Type的所有JSON事件文件。
    返回 list of (features, label, meta_info)
    """
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if limit:
        json_files = json_files[:limit]
    
    results = []
    errors = 0
    stock_cache = {}  # 股票代码 → DataFrame 缓存
    
    for json_file in json_files:
        code = os.path.basename(json_file).replace(".json", "")
        
        # 加载原始股票数据
        if code not in stock_cache:
            df = load_stock_csv(code)
            if df is None:
                errors += 1
                continue
            stock_cache[code] = df
        
        df = stock_cache[code]
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            errors += 1
            continue
        
        # 处理每个time_point事件
        time_points = data.get("time_point", [])
        for tp in time_points:
            try:
                # T日信息
                t_info = tp.get("info", {})
                t_date_str = parse_date(t_info.get("date"))
                if not t_date_str:
                    continue
                t_date = pd.to_datetime(t_date_str)
                
                # d_profit标签
                d_profit = float(tp.get(LABEL_FIELD, 0.0))
                
                # 提取T-60~T-1日特征 (8维)
                features = extract_features(df, t_date, seq_len=SEQ_LEN)
                if features is None:
                    errors += 1
                    continue
                
                # 添加type_id作为第9维特征 (Type-1=0, Type-2=1, Type-3=2)
                type_id_map = {"Type-1": 0.0, "Type-2": 1.0, "Type-3": 2.0}
                type_id_val = type_id_map.get(type_name, 0.0)
                features[:, 8] = type_id_val  # 第9维全设为type_id
                
                meta = {
                    "code": code,
                    "t_date": t_date_str,
                    "d_profit": d_profit,
                    "type": type_name,
                    "d_x": tp.get("setting", {}).get("D_x", 0),
                }
                
                results.append((features, d_profit, meta))
                
            except Exception:
                errors += 1
                continue
    
    print(f"  [{type_name}] 成功: {len(results)}, 失败: {errors}, 总计: {len(results)+errors}")
    return results


def build_hdf5(all_data, output_path, test_ratio=0.3):
    """
    将数据划分为train/test并写入HDF5。
    all_data: list of (features, label, meta)
    按时间排序，前(1-test_ratio)为训练集，后test_ratio为测试集
    """
    # 按T日日期排序（避免未来数据泄露）
    all_data.sort(key=lambda x: x[2]["t_date"])
    
    split_idx = int(len(all_data) * (1 - test_ratio))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    def write_split(data, f, group_name):
        n = len(data)
        features = np.array([d[0] for d in data], dtype=np.float32)  # (n, 60, 9)
        labels = np.array([d[1] for d in data], dtype=np.float32)    # (n,)
        
        grp = f.create_group(group_name)
        grp.create_dataset("features", data=features, compression="gzip", compression_opts=3)
        grp.create_dataset("labels", data=labels, compression="gzip", compression_opts=3)
        
        # 保存meta信息（JSON字符串形式）
        meta_list = [json.dumps(d[2], ensure_ascii=False) for d in data]
        grp.create_dataset("meta", data=np.array(meta_list, dtype=h5py.special_dtype(vlen=str)))
        
        return n
    
    with h5py.File(output_path, "w") as f:
        train_n = write_split(train_data, f, "train")
        test_n = write_split(test_data, f, "test")
    
    return train_n, test_n, split_idx


def compute_summary(all_data, split_idx):
    """计算数据集统计摘要"""
    all_data.sort(key=lambda x: x[2]["t_date"])
    train = all_data[:split_idx]
    test = all_data[split_idx:]
    
    def stats(data):
        labels = np.array([d[1] for d in data])
        types = [d[2]["type"] for d in data]
        dates = [d[2]["t_date"] for d in data]
        
        pos_rate = np.mean(labels > 0) * 100
        return {
            "count": len(data),
            "label_mean": float(np.mean(labels)),
            "label_std": float(np.std(labels)),
            "label_min": float(np.min(labels)),
            "label_max": float(np.max(labels)),
            "pos_rate": float(pos_rate),
            "date_range": [dates[0], dates[-1]] if dates else [],
            "type_breakdown": {t: types.count(t) for t in set(types)},
        }
    
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seq_len": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "total_count": len(all_data),
        "train": stats(train),
        "test": stats(test),
        "split_ratio": {"train": 1-0.3, "test": 0.3},
    }


# ============================================================
# 主流程
# ============================================================

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ---------- Phase 1: 加载所有Type数据 ----------
    print("=" * 60)
    print("Phase 1: 加载 Type-1/2/3 JSON 事件")
    print("=" * 60)
    
    all_data = []
    type_filter = [f"Type-{args.type}"] if args.type in [1, 2, 3] else ["Type-1", "Type-2", "Type-3"]
    
    for type_name in type_filter:
        json_dir = TYPE_DIRS.get(type_name)
        if not os.path.exists(json_dir):
            print(f"  [WARN] 目录不存在: {json_dir}")
            continue
        
        count = len(glob.glob(os.path.join(json_dir, "*.json")))
        print(f"\n处理 {type_name}: {count} 个JSON文件")
        
        results = process_type_events(type_name, json_dir, limit=args.limit)
        all_data.extend(results)
    
    print(f"\n总计加载: {len(all_data)} 条有效样本")
    
    if len(all_data) == 0:
        print("[ERROR] 没有加载到任何有效样本！")
        return
    
    # ---------- Phase 2: 生成HDF5 ----------
    if not args.verify:
        print("\n" + "=" * 60)
        print("Phase 2: 生成 train/test HDF5")
        print("=" * 60)
        
        train_path = os.path.join(OUTPUT_DIR, "train_data.h5")
        test_path = os.path.join(OUTPUT_DIR, "test_data.h5")
        
        train_n, test_n, split_idx = build_hdf5(all_data, train_path, test_ratio=0.3)
        print(f"  训练集: {train_n} 条 -> {train_path}")
        print(f"  测试集: {test_n} 条 -> {test_path}")
        
        # ---------- Phase 3: 输出摘要 ----------
        print("\n" + "=" * 60)
        print("Phase 3: 生成数据摘要")
        print("=" * 60)
        
        summary = compute_summary(all_data, split_idx)
        summary_path = os.path.join(OUTPUT_DIR, "data_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  摘要已保存: {summary_path}")
        
        print("\n--- 数据集摘要 ---")
        print(f"  总样本: {summary['total_count']}")
        print(f"  训练集: {summary['train']['count']} 条, d_profit均值: {summary['train']['label_mean']:.4f}, "
              f"正收益占比: {summary['train']['pos_rate']:.2f}%")
        print(f"  测试集: {summary['test']['count']} 条, d_profit均值: {summary['test']['label_mean']:.4f}, "
              f"正收益占比: {summary['test']['pos_rate']:.2f}%")
        print(f"  时间范围: {summary['train']['date_range'][0]} ~ {summary['test']['date_range'][1]}")
        print(f"  Type分布: {summary['train']['type_breakdown']}")
    
    print("\n✅ 数据准备完成！")
    print(f"\n下一步: python lstm_attention_model.py --mode train")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Stage2 数据准备")
    parser.add_argument("--verify", action="store_true", help="仅验证数据，不生成HDF5")
    parser.add_argument("--type", type=int, choices=[1, 2, 3], default=None,
                        help="仅处理指定Type（1/2/3），默认处理全部")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制每个Type处理的JSON文件数量（用于快速测试）")
    
    args = parser.parse_args()
    main(args)
