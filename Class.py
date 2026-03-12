import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ta  # 技术指标库


# 假设您的日线数据CSV结构如下：
# stock_id, date, open, high, low, close, volume, recommendation_date, ideal_return_rate
# 示例数据形状：每条记录包含前N天的日线数据

class StockDataProcessor:
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
        self.scaler = StandardScaler()

    def load_and_process_data(self, csv_path):
        """
        加载并处理日线数据
        """
        # 读取CSV数据
        df = pd.read_csv(csv_path)

        # 确保数据按股票和时间排序
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['stock_id', 'date'])

        return df

    def calculate_technical_indicators(self, df):
        """
        计算技术指标
        """
        # 分组计算每个股票的技术指标
        technical_data = []

        for stock_id, group in df.groupby('stock_id'):
            group = group.sort_values('date')

            # 价格数据
            prices = group[['open', 'high', 'low', 'close', 'volume']].copy()

            # 计算技术指标
            # 趋势指标
            prices['sma_5'] = ta.trend.sma_indicator(prices['close'], window=5)
            prices['sma_10'] = ta.trend.sma_indicator(prices['close'], window=10)
            prices['sma_20'] = ta.trend.sma_indicator(prices['close'], window=20)
            prices['ema_12'] = ta.trend.ema_indicator(prices['close'], window=12)
            prices['ema_26'] = ta.trend.ema_indicator(prices['close'], window=26)

            # MACD
            macd = ta.trend.MACD(prices['close'])
            prices['macd'] = macd.macd()
            prices['macd_signal'] = macd.macd_signal()
            prices['macd_histogram'] = macd.macd_diff()

            # RSI
            prices['rsi_14'] = ta.momentum.rsi(prices['close'], window=14)
            prices['rsi_28'] = ta.momentum.rsi(prices['close'], window=28)

            # 波动率指标
            prices['atr_14'] = ta.volatility.average_true_range(
                prices['high'], prices['low'], prices['close'], window=14
            )

            # 支撑阻力
            prices['bb_upper'] = ta.volatility.bollinger_hband(prices['close'])
            prices['bb_lower'] = ta.volatility.bollinger_lband(prices['close'])
            prices['bb_middle'] = ta.volatility.bollinger_mavg(prices['close'])

            # 成交量指标
            prices['volume_sma'] = ta.volume.volume_sma(prices['volume'], window=10)
            prices['obv'] = ta.volume.on_balance_volume(prices['close'], prices['volume'])

            # 添加股票ID和日期
            prices['stock_id'] = stock_id
            prices['date'] = group['date'].values

            technical_data.append(prices)

        technical_df = pd.concat(technical_data, ignore_index=True)
        return technical_df

    def create_sequences(self, df, target_df):
        """
        创建序列数据：每个推荐点取前N天的数据作为特征
        target_df包含：stock_id, recommendation_date, ideal_return_rate, target_score
        """
        sequences = []
        targets = []

        for _, target_row in target_df.iterrows():
            stock_id = target_row['stock_id']
            rec_date = target_row['recommendation_date']

            # 获取该股票推荐日期前lookback_days的数据
            stock_data = df[df['stock_id'] == stock_id].copy()
            stock_data = stock_data[stock_data['date'] <= rec_date].tail(self.lookback_days)

            if len(stock_data) >= self.lookback_days:
                # 选择特征列
                feature_cols = ['open', 'high', 'low', 'close', 'volume',
                                'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
                                'macd', 'macd_signal', 'macd_histogram',
                                'rsi_14', 'rsi_28', 'atr_14',
                                'bb_upper', 'bb_lower', 'bb_middle',
                                'volume_sma', 'obv']

                # 只选择存在的列
                available_cols = [col for col in feature_cols if col in stock_data.columns]

                sequence = stock_data[available_cols].values
                target = target_row['target_score']

                sequences.append(sequence)
                targets.append(target)

        return np.array(sequences), np.array(targets)

    def prepare_target_variable(self, ideal_returns):
        """
        根据理想收益率分布映射到0-1评分
        参考提供的分布统计表进行更合理的映射
        """
        # 使用分位数映射，避免极端值影响
        returns = np.array(ideal_returns)

        # 基于分布统计的映射策略
        # 可以根据xlsx中的分布区间调整权重
        q_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantiles = np.quantile(returns, q_values)

        # 创建分段映射函数
        def map_return_to_score(return_rate):
            if return_rate <= quantiles[0]:  # 最低1%
                return 0.0
            elif return_rate >= quantiles[-1]:  # 最高1%
                return 1.0
            else:
                # 线性插值
                for i in range(len(quantiles) - 1):
                    if quantiles[i] <= return_rate < quantiles[i + 1]:
                        t = (return_rate - quantiles[i]) / (quantiles[i + 1] - quantiles[i])
                        return q_values[i] + t * (q_values[i + 1] - q_values[i])
                return 0.5

        scores = np.array([map_return_to_score(r) for r in returns])
        return scores


# 使用示例
processor = StockDataProcessor(lookback_days=30)

# 加载日线数据
daily_data = processor.load_and_process_data('your_stock_data.csv')

# 计算技术指标
technical_data = processor.calculate_technical_indicators(daily_data)

# 准备目标变量（假设您有包含理想收益率的目标数据）
target_data = pd.DataFrame({
    'stock_id': daily_data['stock_id'].unique()[:1000],  # 示例
    'recommendation_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
    'ideal_return_rate': np.random.normal(0.02, 0.1, 1000)  # 替换为实际数据
})

# 映射收益率到评分
target_data['target_score'] = processor.prepare_target_variable(
    target_data['ideal_return_rate']
)

# 创建序列数据
sequences, targets = processor.create_sequences(technical_data, target_data)

print(f"序列数据形状: {sequences.shape}")  # (样本数, 时间步长, 特征数)
print(f"目标变量形状: {targets.shape}")