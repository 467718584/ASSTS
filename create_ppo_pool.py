"""
打包分钟数据股票池
根据模型筛选结果，打包对应的分钟级数据

优化：
1. 只获取分钟数据覆盖范围内的推荐日（2025-07-25 ~ 2025-10-09）
2. 每个推荐日只获取当天及之后两天的分钟数据
"""
import os
import json
import shutil
from datetime import datetime, timedelta

# 配置
MIN_DATA_DIR = '/home/zzy/project/ASSTS/data_min'  # 分钟级数据根目录
POOL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
FILTER_RESULT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'

# 分钟数据日期范围
MIN_DATA_START = '2025-07-25'
MIN_DATA_END = '2025-10-09'

def load_filter_results():
    """加载筛选结果"""
    results = {}
    # 加载所有6个模型的结果 - 统一阈值 0.5
    model_names = ['30d_2s3h', '30d_2s3e', '60d_2s3h', '60d_2s3e', '120d_2s3h', '120d_2s3e']
    THRESHOLD = '0.5'
    
    for model_name in model_names:
        file_path = os.path.join(FILTER_RESULT_DIR, f'stocks_by_{model_name}_threshold_{THRESHOLD}.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if len(data) > 0:
                results[model_name] = data
                print(f"加载 {model_name}: {len(data)} 条记录")
            else:
                print(f"跳过空文件: {file_path}")
        else:
            print(f"文件不存在: {file_path}")
    
    return results

def get_available_dates_for_stock(code):
    """获取某只股票可用的分钟数据日期（统一格式：YYYY-MM-DD）"""
    stock_dir = os.path.join(MIN_DATA_DIR, code)
    if not os.path.exists(stock_dir):
        return set()
    
    dates = set()
    for f in os.listdir(stock_dir):
        if f.endswith('.csv'):
            date_str = f.replace('.csv', '')
            # 统一格式：确保是 YYYY-MM-DD 格式
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                date_str = dt.strftime('%Y-%m-%d')
            except:
                pass
            dates.add(date_str)
    return dates

def get_target_dates(recommend_date, available_dates):
    """
    获取目标日期列表：推荐日及之后两天
    只返回实际存在于分钟数据中的日期
    """
    # 统一日期格式：处理 2025-10-9 -> 2025-10-09
    try:
        recommend_dt = datetime.strptime(recommend_date, '%Y-%m-%d')
        recommend_date = recommend_dt.strftime('%Y-%m-%d')
    except:
        return []
    
    target_dates = []
    for i in range(3):  # 推荐日当天及之后2天
        target_dt = recommend_dt + timedelta(days=i)
        target_str = target_dt.strftime('%Y-%m-%d')
        if target_str in available_dates:
            target_dates.append(target_str)
    
    return target_dates

def create_pool_by_model(results):
    """为每个模型创建对应的分钟数据股票池"""
    os.makedirs(POOL_OUTPUT_DIR, exist_ok=True)
    
    # 将分钟数据日期范围转为集合便于快速查找
    min_start = datetime.strptime(MIN_DATA_START, '%Y-%m-%d')
    min_end = datetime.strptime(MIN_DATA_END, '%Y-%m-%d')
    
    for model_name, stocks in results.items():
        print(f"\n处理模型: {model_name}")
        print(f"  原始记录数: {len(stocks)}")
        
        # 过滤：在分钟数据日期范围内的推荐日
        valid_stocks = []
        for stock in stocks:
            date = stock['date']
            try:
                date_dt = datetime.strptime(date, '%Y-%m-%d')
                if min_start <= date_dt <= min_end:
                    valid_stocks.append(stock)
            except:
                continue
        
        print(f"  分钟数据范围内的记录数: {len(valid_stocks)}")
        
        # 按股票代码分组
        stock_pool = {}
        for stock in valid_stocks:
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
        
        # 复制对应股票的分钟数据（只复制目标日期）
        copied_count = 0
        actual_stock_count = 0
        incomplete_count = 0
        
        for code, date_list in stock_pool.items():
            # 获取该股票可用的分钟数据日期
            available_dates = get_available_dates_for_stock(code)
            if not available_dates:
                continue
            
            # 收集所有需要获取的日期（去重）
            all_target_dates = set()
            for item in date_list:
                target_dates = get_target_dates(item['date'], available_dates)
                all_target_dates.update(target_dates)
            
            if not all_target_dates:
                continue
            
            # 【关键检查】必须有至少3天数据
            # Day1: 推荐日，Day2: 开盘买，Day3: 卖出
            if len(all_target_dates) < 3:
                incomplete_count += 1
                continue
            
            # 创建股票目录
            dest_dir = os.path.join(pool_dir, code)
            os.makedirs(dest_dir, exist_ok=True)
            
            # 复制目标日期的分钟数据
            for target_date in all_target_dates:
                src_file = os.path.join(MIN_DATA_DIR, code, f'{target_date}.csv')
                dest_file = os.path.join(dest_dir, f'{target_date}.csv')
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dest_file)
                    copied_count += 1
            
            actual_stock_count += 1
        
        print(f"  数据不完整跳过: {incomplete_count}")
        print(f"  实际复制股票数(2天+): {actual_stock_count}")
        print(f"  复制文件数: {copied_count}")
        
        # 保存股票池元信息
        meta_file = os.path.join(pool_dir, 'meta.json')
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_name,
                'original_records': len(stocks),
                'valid_records_in_min_range': len(valid_stocks),
                'stock_count': actual_stock_count,
                'date_range': f'{MIN_DATA_START} to {MIN_DATA_END}',
                'data_per_stock': '推荐日及之后2天',
                'stocks': {code: date_list for code, date_list in list(stock_pool.items())[:10]}
            }, f, ensure_ascii=False, indent=2)
    
    print(f"\n股票池创建完成!")

def create_combined_pool(results):
    """创建综合股票池（所有模型筛选结果的并集）"""
    # 收集所有在分钟数据范围内的股票
    all_valid_codes = set()
    all_stock_dates = {}  # code -> [(date, threshold, model), ...]
    
    min_start = datetime.strptime(MIN_DATA_START, '%Y-%m-%d')
    min_end = datetime.strptime(MIN_DATA_END, '%Y-%m-%d')
    
    for model_name, stocks in results.items():
        for stock in stocks:
            code = stock['code'].replace('.json', '')
            date = stock['date']
            threshold = stock['threshold']
            
            # 检查日期是否在分钟数据范围内
            try:
                date_dt = datetime.strptime(date, '%Y-%m-%d')
                if not (min_start <= date_dt <= min_end):
                    continue
            except:
                continue
            
            all_valid_codes.add(code)
            
            if code not in all_stock_dates:
                all_stock_dates[code] = []
            all_stock_dates[code].append({
                'date': date,
                'threshold': threshold,
                'model': model_name
            })
    
    print(f"\n综合股票池（分钟数据范围内）唯一股票数: {len(all_valid_codes)}")
    
    pool_dir = os.path.join(POOL_OUTPUT_DIR, 'ppo_pool_combined_min1')
    os.makedirs(pool_dir, exist_ok=True)
    
    copied_count = 0
    actual_stock_count = 0
    
    for code, date_list in all_stock_dates.items():
        available_dates = get_available_dates_for_stock(code)
        if not available_dates:
            continue
        
        # 收集所有目标日期
        all_target_dates = set()
        for item in date_list:
            target_dates = get_target_dates(item['date'], available_dates)
            all_target_dates.update(target_dates)
        
        if not all_target_dates:
            continue
        
        dest_dir = os.path.join(pool_dir, code)
        os.makedirs(dest_dir, exist_ok=True)
        
        for target_date in all_target_dates:
            src_file = os.path.join(MIN_DATA_DIR, code, f'{target_date}.csv')
            dest_file = os.path.join(dest_dir, f'{target_date}.csv')
            if os.path.exists(src_file):
                shutil.copy2(src_file, dest_file)
                copied_count += 1
        
        actual_stock_count += 1
    
    print(f"综合股票池实际复制股票数: {actual_stock_count}")
    print(f"综合股票池复制文件数: {copied_count}")
    
    # 保存元信息
    meta_file = os.path.join(pool_dir, 'meta.json')
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'combined_all_models',
            'stock_count': actual_stock_count,
            'date_range': f'{MIN_DATA_START} to {MIN_DATA_END}',
            'data_per_stock': '推荐日及之后2天',
            'models_included': list(results.keys())
        }, f, ensure_ascii=False, indent=2)

def main():
    print("=" * 60)
    print("PPO股票池创建 (优化版)")
    print(f"分钟数据范围: {MIN_DATA_START} ~ {MIN_DATA_END}")
    print("每个推荐日获取: 当天 + 之后2天")
    print("=" * 60)
    
    # 加载筛选结果
    results = load_filter_results()
    
    # 为每个模型创建股票池
    create_pool_by_model(results)
    
    # 创建综合股票池
    create_combined_pool(results)
    
    print("\n" + "=" * 60)
    print("=== 股票池创建完成 ===")
    print(f"输出目录: {POOL_OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
