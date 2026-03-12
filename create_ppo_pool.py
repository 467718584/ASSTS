"""
打包分钟数据股票池
根据模型筛选结果，打包对应的分钟级数据
"""
import os
import json
import shutil
import pandas as pd

# 配置
MIN_DATA_DIR = '/home/zzy/project/ASSTS/data_min'  # 分钟级数据根目录
POOL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
FILTER_RESULT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'

def load_filter_results():
    """加载筛选结果"""
    results = {}
    # 加载所有模型的结果
    model_names = ['30d_2s3h', '60d_2s3e', '60d_2s3h', '120d_2s3h']
    
    for model_name in model_names:
        file_path = os.path.join(FILTER_RESULT_DIR, f'stocks_by_{model_name}_threshold_0.5.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                results[model_name] = json.load(f)
            print(f"加载 {model_name}: {len(results[model_name])} 条记录")
        else:
            print(f"文件不存在: {file_path}")
    
    return results

def get_unique_codes(results):
    """获取所有唯一的股票代码"""
    all_codes = set()
    for model_name, stocks in results.items():
        for stock in stocks:
            # 去除.json后缀
            code = stock['code'].replace('.json', '')
            all_codes.add(code)
    return all_codes

def create_pool_by_model(results, min_date='2025-08-01', max_date='2025-09-30'):
    """为每个模型创建对应的分钟数据股票池"""
    os.makedirs(POOL_OUTPUT_DIR, exist_ok=True)
    
    for model_name, stocks in results.items():
        print(f"\n处理模型: {model_name}")
        
        # 获取该模型筛选出的股票代码和日期
        stock_pool = {}
        for stock in stocks:
            code = stock['code'].replace('.json', '')
            date = stock['date']
            threshold = stock['threshold']
            
            if code not in stock_pool:
                stock_pool[code] = []
            stock_pool[code].append({
                'date': date,
                'threshold': threshold
            })
        
        print(f"  唯一股票数: {len(stock_pool)}")
        
        # 创建股票池目录
        pool_dir = os.path.join(POOL_OUTPUT_DIR, f'ppo_pool_{model_name}_min1')
        os.makedirs(pool_dir, exist_ok=True)
        
        # 复制对应股票的分钟数据
        copied_count = 0
        for code, dates in stock_pool.items():
            stock_dir = os.path.join(MIN_DATA_DIR, code)
            if os.path.exists(stock_dir):
                # 创建股票目录
                dest_dir = os.path.join(pool_dir, code)
                os.makedirs(dest_dir, exist_ok=True)
                
                # 复制该股票所有分钟数据
                for csv_file in os.listdir(stock_dir):
                    if csv_file.endswith('.csv'):
                        src_file = os.path.join(stock_dir, csv_file)
                        dest_file = os.path.join(dest_dir, csv_file)
                        shutil.copy2(src_file, dest_file)
                        copied_count += 1
        
        print(f"  复制文件数: {copied_count}")
        
        # 保存股票池元信息
        meta_file = os.path.join(pool_dir, 'meta.json')
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_name,
                'stock_count': len(stock_pool),
                'date_range': f'{min_date} to {max_date}',
                'stocks': {code: dates for code, dates in list(stock_pool.items())[:100]}  # 只保留前100个作为示例
            }, f, ensure_ascii=False, indent=2)
    
    print(f"\n股票池创建完成!")

def create_combined_pool(results):
    """创建综合股票池（所有模型筛选结果的并集）"""
    all_codes = get_unique_codes(results)
    print(f"\n综合股票池唯一股票数: {len(all_codes)}")
    
    pool_dir = os.path.join(POOL_OUTPUT_DIR, 'ppo_pool_combined_min1')
    os.makedirs(pool_dir, exist_ok=True)
    
    copied_count = 0
    for code in all_codes:
        stock_dir = os.path.join(MIN_DATA_DIR, code)
        if os.path.exists(stock_dir):
            dest_dir = os.path.join(pool_dir, code)
            os.makedirs(dest_dir, exist_ok=True)
            
            for csv_file in os.listdir(stock_dir):
                if csv_file.endswith('.csv'):
                    src_file = os.path.join(stock_dir, csv_file)
                    dest_file = os.path.join(dest_dir, csv_file)
                    shutil.copy2(src_file, dest_file)
                    copied_count += 1
    
    print(f"综合股票池复制文件数: {copied_count}")
    
    # 保存元信息
    meta_file = os.path.join(pool_dir, 'meta.json')
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'combined_all_models',
            'stock_count': len(all_codes),
            'date_range': '2025-08-01 to 2025-09-30',
            'models_included': list(results.keys())
        }, f, ensure_ascii=False, indent=2)

def main():
    # 加载筛选结果
    results = load_filter_results()
    
    # 为每个模型创建股票池
    create_pool_by_model(results)
    
    # 创建综合股票池
    create_combined_pool(results)
    
    print("\n=== 股票池创建完成 ===")
    print(f"输出目录: {POOL_OUTPUT_DIR}")

if __name__ == '__main__':
    main()
