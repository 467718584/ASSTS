"""
股票预测筛选脚本 - 分批处理版本
对6个模型进行预测，筛选阈值>0.5的股票
"""
import os
import pickle
import json
import numpy as np
import torch

# 添加项目路径
import sys
sys.path.append('/home/zzy/project/ASSTS/ASSTS-stock_class')

from predict_model import StockPredictor

# 配置
MODEL_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/model_file'
DATASET_PATH = '/home/zzy/project/ASSTS/ASSTS-stock_class/dataset/0d-m-data-120d-preprocess-14f-2s3e-class-003-TEST.pkl'
OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
THRESHOLD = 0.5
BATCH_SIZE = 1000  # 分批处理

# 模型列表 - 6个预训练模型 (使用实际存在的模型文件)
MODELS = {
    '30d_2s3e': {'file': '0d_30d_14f_2s3e_BCE_0005-003_AUC_05437.pth', 'length': 30},
    '30d_2s3h': {'file': '0d_30d_14f_2s3h_BCE_0005-003_AUC_05844.pth', 'length': 30},
    '60d_2s3e': {'file': '0d_60d_14f_2s3e_BCE_0005-003_AUC_05343.pth', 'length': 60},
    '60d_2s3h': {'file': '0d_60d_14f_2s3h_BCE_0005-003_AUC_05736.pth', 'length': 60},
    '120d_2s3e': {'file': '0d_120d_14f_2s3e_BCE_0005-003_AUC_05434.pth', 'length': 120},
    '120d_2s3h': {'file': '0d_120d_14f_2s3h_BCE_0005-003_AUC_05843.pth', 'length': 120},
}

def load_dataset():
    """加载测试数据集"""
    print(f"加载数据集: {DATASET_PATH}")
    with open(DATASET_PATH, 'rb') as f:
        data_pkl = pickle.load(f)
    print(f"数据集keys: {data_pkl.keys()}")
    print(f"特征数据shape: {data_pkl['features_data'].shape}")
    print(f"标签数据shape: {data_pkl['label_list'].shape}")
    return data_pkl

def predict_with_model(model_name, model_info, data_pkl):
    """使用指定模型进行预测 - 分批处理"""
    model_file = model_info['file']
    feature_length = model_info['length']
    
    model_path = os.path.join(MODEL_DIR, model_file)
    print(f"\n{'='*50}")
    print(f"预测模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"特征长度: {feature_length}")
    
    # 初始化预测器
    try:
        predictor = StockPredictor(model_path, input_channels=14, time_steps=feature_length)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return []
    
    # 获取数据
    features_data = data_pkl['features_data']
    info_list = data_pkl['info_list']
    
    # 提取对应时间步长的特征
    if feature_length == 30:
        features = features_data[:, -30:, :]
    elif feature_length == 60:
        features = features_data[:, -60:, :]
    else:  # 120
        features = features_data
    
    total_samples = features.shape[0]
    print(f"输入特征shape: {features.shape}")
    print(f"总样本数: {total_samples}")
    
    # 分批预测
    results = []
    for batch_start in range(0, total_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_samples)
        batch_features = features[batch_start:batch_end]
        batch_info = info_list[batch_start:batch_end]
        
        classes, probabilities = predictor.predict(batch_features)
        
        # 筛选阈值>0.5的股票
        for i, (cls, prob) in enumerate(zip(classes, probabilities)):
            if prob > THRESHOLD:
                info = batch_info[i]
                results.append({
                    'code': info['code'],
                    'date': info['date'],
                    'threshold': float(prob),
                    'class': int(cls)
                })
        
        if (batch_start // BATCH_SIZE) % 10 == 0:
            print(f"  处理进度: {batch_end}/{total_samples} ({100*batch_end/total_samples:.1f}%)")
    
    print(f"筛选结果: 共 {len(results)} 只股票阈值 > {THRESHOLD}")
    return results

def save_results(model_name, results):
    """保存筛选结果"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f'stocks_by_{model_name}_threshold_{THRESHOLD}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存: {output_file}")
    return output_file

def main():
    # 加载数据
    data_pkl = load_dataset()
    
    all_results = {}
    
    # 对每个模型进行预测
    for model_key, model_info in MODELS.items():
        results = predict_with_model(model_key, model_info, data_pkl)
        all_results[model_key] = results
        
        # 保存每个模型的单独结果
        save_results(model_key, results)
        
        # 打印前5个示例
        print(f"\n{model_key} 筛选结果示例(前5个):")
        for r in results[:5]:
            print(f"  {r['code']} | {r['date']} | 阈值: {r['threshold']:.4f}")
    
    # 保存汇总结果
    summary_file = os.path.join(OUTPUT_DIR, 'all_models_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print("所有模型预测完成!")
    print(f"汇总结果: {summary_file}")
    
    # 打印统计
    for model_key, results in all_results.items():
        print(f"{model_key}: {len(results)} 只股票")

if __name__ == '__main__':
    main()
