#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 LSTM+Attention 数据准备 - 按股票分组高效版

策略:
1. 先按stock分组所有事件
2. 每只股票只读一次CSV，向量化提取序列
3. 流式写入HDF5，内存占用 ~100MB
"""

import os
import json
import glob
import time
import warnings
import gc

import numpy as np
import pandas as pd
import h5py

warnings.filterwarnings("ignore")

BASE_DIR = "/home/zzy/project/ASSTS/ASSTS-stock_class"
STOCK_DATA_DIR = "/home/zzy/project/ASSTS/stock_dir/tushare_stock_1107-all"
STRATEGY1_DIR = "/tmp/strategy1"
OUTPUT_DIR = os.path.join(BASE_DIR, "lstm_stage", "data")

SEQ_LEN = 60
FEATURE_DIM = 9
TYPE_ID_MAP = {"Type-1": 0.0, "Type-2": 1.0, "Type-3": 2.0}
LABEL_FIELD = "d_profit"
CHUNK_SIZE = 50000

TYPE_DIRS = {
    "Type-1": os.path.join(STRATEGY1_DIR, "tushare_output_all-0-1~9-new"),
    "Type-2": os.path.join(STRATEGY1_DIR, "tushare_output_all-1-1~9-new"),
    "Type-3": os.path.join(STRATEGY1_DIR, "tushare_output_all-2-1~9-new"),
}

def parse_date(date_val):
    if isinstance(date_val, (int, float)):
        s = str(int(date_val))
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    elif isinstance(date_val, str):
        return date_val[:10]
    return None

def load_stock_csv_fast(code):
    """快速加载股票CSV，返回标准化的DataFrame"""
    csv_path = os.path.join(STOCK_DATA_DIR, f"{code}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if '日期' in c or cl == 'date':
                col_map[c] = 'date'
            elif '开盘' in c or cl == 'open':
                col_map[c] = 'open'
            elif '最高' in c or cl == 'high':
                col_map[c] = 'high'
            elif '最低' in c or cl == 'low':
                col_map[c] = 'low'
            elif '收盘' in c or cl == 'close':
                col_map[c] = 'close'
            elif '成交量' in c or cl == 'volume':
                col_map[c] = 'volume'
            elif 'diff' in cl:
                col_map[c] = 'diff'
            elif 'dea' in cl:
                col_map[c] = 'dea'
        if col_map:
            df = df.rename(columns=col_map)
        
        if 'date' not in df.columns:
            return None
        
        # 统一日期格式
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        df = df.sort_values('date').reset_index(drop=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'diff', 'dea']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = 0.0
        
        # 构建日期→行索引的映射（加速查找）
        df['_date_idx'] = range(len(df))
        date_to_idx = dict(zip(df['date'], df['_date_idx']))
        df['_date_idx'] = df['date'].map(date_to_idx)
        
        return df
    except Exception:
        return None

def extract_sequences_vectorized(df, events_for_stock):
    """
    向量化提取多序列：df全量数据 + events列表
    返回 (features, labels, metas, n_valid)
    """
    if df is None or len(events_for_stock) == 0:
        return None, None, None, 0
    
    n = len(events_for_stock)
    features = np.zeros((n, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)
    metas = []
    valid_count = 0
    
    # 预计算归一化参数（对整个df做，减少重复计算）
    feat_cols = ['open', 'high', 'low', 'close', 'volume', 'diff', 'dea']  # 7维 + market_return(0) + type_id = 9
    feat_data = df[feat_cols].values  # (n_rows, 7)
    
    # 全局归一化
    mean = np.mean(feat_data, axis=0, keepdims=True)
    std = np.std(feat_data, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    
    date_to_idx = dict(zip(df['date'], range(len(df))))
    
    for i, evt in enumerate(events_for_stock):
        t_date = evt['t_date']
        if t_date not in date_to_idx:
            continue
        
        idx = date_to_idx[t_date]
        start_idx = max(0, idx - SEQ_LEN)
        end_idx = idx
        
        if end_idx - start_idx < SEQ_LEN:
            continue
        
        # 取窗口数据
        window = feat_data[start_idx:end_idx]  # (60, 7)
        
        # 归一化（用全局参数）
        window_norm = (window - mean) / std  # (60, 7)
        
        # 添加type_id作为第9维
        type_id = TYPE_ID_MAP.get(evt['type'], 0.0)
        # 添加market_return=0作为第8维，type_id作为第9维
        market_col = np.zeros((SEQ_LEN, 1), dtype=np.float32)  # 第8维: 大盘收益率(暂时设为0)
        type_col = np.full((SEQ_LEN, 1), type_id, dtype=np.float32)  # 第9维: type_id
        seq9 = np.hstack([window_norm, market_col, type_col])  # (60, 9)
        
        features[valid_count] = seq9
        labels[valid_count] = evt['d_profit']
        metas.append(json.dumps({
            'code': evt['code'], 't_date': t_date,
            'type': evt['type'], 'd_x': evt['d_x'], 'd_profit': evt['d_profit']
        }, ensure_ascii=False))
        valid_count += 1
    
    return features[:valid_count], labels[:valid_count], metas, valid_count

# ============================================================
# 主流程
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: 收集所有事件并按股票分组")
print("=" * 60)

# 按股票分组
events_by_stock = {}  # code -> list of events
all_events = []

for type_name, tdir in TYPE_DIRS.items():
    json_files = glob.glob(os.path.join(tdir, "*.json"))
    print(f"  {type_name}: {len(json_files)} 个JSON")
    
    for fpath in json_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            code = os.path.basename(fpath).replace('.json', '')
            
            for tp in data.get('time_point', []):
                t_date_str = parse_date(tp['info']['date'])
                if not t_date_str:
                    continue
                
                evt = {
                    'code': code, 't_date': t_date_str,
                    'd_profit': float(tp.get(LABEL_FIELD, 0.0)),
                    'type': type_name,
                    'd_x': tp.get('setting', {}).get('D_x', 0),
                }
                
                if code not in events_by_stock:
                    events_by_stock[code] = []
                events_by_stock[code].append(evt)
                all_events.append(evt)
        except Exception:
            pass

print(f"  总事件: {len(all_events)}, 涉及股票: {len(events_by_stock)}")

# 按时间排序划分
all_events.sort(key=lambda x: x['t_date'])
split_idx = int(len(all_events) * 0.7)
train_events_set = set(e['code'] + '|' + e['t_date'] for e in all_events[:split_idx])
test_events_set = set(e['code'] + '|' + e['t_date'] for e in all_events[split_idx:])

# 按股票分组后的训练/测试事件
train_by_stock = {}  # code -> list
test_by_stock = {}   # code -> list

for code, evts in events_by_stock.items():
    for evt in evts:
        key = evt['code'] + '|' + evt['t_date']
        if key in train_events_set:
            if code not in train_by_stock:
                train_by_stock[code] = []
            train_by_stock[code].append(evt)
        else:
            if code not in test_by_stock:
                test_by_stock[code] = []
            test_by_stock[code].append(evt)

print(f"  训练集: {sum(len(v) for v in train_by_stock.values())} 条, 股票: {len(train_by_stock)}")
print(f"  测试集: {sum(len(v) for v in test_by_stock.values())} 条, 股票: {len(test_by_stock)}")

del events_by_stock, all_events
gc.collect()

# ============================================================
# Step 2: 流式处理写入
# ============================================================
def stream_write(by_stock, output_path, desc=""):
    """按股票分组处理，流式写入HDF5"""
    all_features = []
    all_labels = []
    all_metas = []
    total_valid = 0
    processed = 0
    total_stocks = len(by_stock)
    
    for code, evts in by_stock.items():
        # 加载股票CSV
        df = load_stock_csv_fast(code)
        if df is None:
            continue
        
        # 提取所有序列
        feats, labs, metas, n_valid = extract_sequences_vectorized(df, evts)
        
        if n_valid > 0:
            all_features.append(feats)
            all_labels.append(labs)
            all_metas.extend(metas)
            total_valid += n_valid
        
        processed += 1
        if processed % 500 == 0:
            print(f"  {desc} {processed}/{total_stocks} ({processed*100//total_stocks}%)")
            gc.collect()
    
    print(f"  {desc} 完成: {processed}只股票, 有效样本{total_valid}")
    
    if total_valid == 0:
        return 0
    
    # 合并写入
    all_feat = np.concatenate(all_features, axis=0)
    all_lab = np.concatenate(all_labels)
    all_meta = np.array(all_metas, dtype=h5py.special_dtype(vlen=str))
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('features', data=all_feat, compression='gzip', compression_opts=3)
        f.create_dataset('labels', data=all_lab, compression='gzip', compression_opts=3)
        f.create_dataset('meta', data=all_meta)
    
    del all_features, all_labels, all_metas, all_feat, all_lab, all_meta
    gc.collect()
    
    return total_valid

print("\n" + "=" * 60)
print("Step 2: 处理训练集")
print("=" * 60)
t0 = time.time()
train_path = os.path.join(OUTPUT_DIR, 'train_data.h5')
train_n = stream_write(train_by_stock, train_path, "训练集")
print(f"  耗时: {time.time()-t0:.1f}s")

print("\n" + "=" * 60)
print("Step 3: 处理测试集")
print("=" * 60)
t0 = time.time()
test_path = os.path.join(OUTPUT_DIR, 'test_data.h5')
test_n = stream_write(test_by_stock, test_path, "测试集")
print(f"  耗时: {time.time()-t0:.1f}s")

# ============================================================
# Step 4: 摘要
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 保存摘要")
print("=" * 60)

summary = {
    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'seq_len': SEQ_LEN,
    'feature_dim': FEATURE_DIM,
    'feature_names': ['open', 'high', 'low', 'close', 'volume', 'diff', 'dea', 'market_return', 'type_id'],
    'total_count': train_n + test_n,
    'train_count': train_n,
    'test_count': test_n,
    'split_ratio': {'train': 0.7, 'test': 0.3},
}

summary_path = os.path.join(OUTPUT_DIR, 'data_summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n✅ 完成!")
print(f"  训练集: {train_n} 条 -> {train_path}")
print(f"  测试集: {test_n} 条 -> {test_path}")
print(f"  摘要: {summary_path}")
print(f"\n下一步: python lstm_attention_model.py --mode train")
