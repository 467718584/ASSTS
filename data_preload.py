import csv
import os
import json
import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Callable, Dict

from pyexpat import features
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.utils.data import DataLoader, TensorDataset


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tools.feature_engineer_f11 import engineer_features_f8, engineer_features_f11
from tools.feature_engineer_f14 import engineer_features_f14


def preload(length=120):
    start_year = 2010
    end_year = 2025
    months = 12
    days = 31
    D_X_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    date_list = {}
    for i in range(start_year, end_year + 1):
        date_list[str(i)] = {}
        for j in range(months):
            date_list[str(i)][str(j + 1)] = {}
            for x in range(days):
                date_list[str(i)][str(j + 1)][str(x + 1)] = {'list': [], 'num': 0}

    data_length = length
    data_num = 0

    f_path = '../tushare_output_dir/tushare_output_all-0-1~9-new'
    csv_dir = '../stock_dir/tushare_stock_1107-all'
    for i in os.listdir(f_path):
        print(os.path.join(f_path, i))
        f = open(os.path.join(f_path, i), 'r')
        t = json.load(f)
        # date_s_list = [x['info']['date'].split('-') for x in t['time_point']]
        # dx_list = [{'D_x': x['setting']['D_x'], 'min': x['info']['min']}
        #            for x in t['time_point']]
        date_s_list = []
        dx_list = []
        csv_path = os.path.join(csv_dir, i.split('.')[0] + '.csv')
        print(csv_path)
        df = pd.read_csv(csv_path)
        # print(csv_reader)
        for x in t['time_point']:
            data_value_list = x['info']['date'].split('-')
            data_value = data_value_list[0] + data_value_list[1] + data_value_list[2]
            # print(df['日期'])
            if x['setting']['D_x'] in D_X_list:
                result_rows = df[df['日期'] == int(data_value)]
                if not result_rows.empty:
                    target_idx = result_rows.index[0]
                    if target_idx > data_length:
                        start_idx = target_idx - data_length + 1
                        sliced_data = df.loc[start_idx: target_idx]

                        # === 关键修复：填充缺失值 ===
                        sliced_data = sliced_data.fillna(method='ffill').fillna(0)

                        # 检查切片后长度是否严格为 120
                        if len(sliced_data) == data_length:
                            date_s_list.append(x['info']['date'].split('-'))
                            x.update({'data': sliced_data})
                            dx_list.append(x)
                            data_num += 1

                        # date_s_list.append(x['info']['date'].split('-'))
                        # x.update({'data': df.loc[target_idx-data_length:target_idx]})
                        # dx_list.append(x)
                        # # print(df.loc[target_idx-data_length:target_idx])
                        # data_num += 1
        for j, d in enumerate(date_s_list):
            date_list[d[0]][str(int(d[1]))][str(int(d[2]))]['list'].append((i, dx_list[j]))

    print(data_num)
    # 1. 将字典保存到 .pkl 文件
    with open('dataset/0d-m-data-150d.pkl', 'wb') as file: # 注意是 'wb' 模式
        pickle.dump(date_list, file)
    print("字典已保存！")


def plot_data_distribution(data, title="数据分布图", save_path=None):
    """
    绘制完整的数据分布分析图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 直方图
    axes[0, 0].hist(data, bins=500, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('直方图')
    axes[0, 0].set_xlabel('数值')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].grid(alpha=0.3)

    # 2. 箱线图
    axes[0, 1].boxplot(data, vert=True, patch_artist=True)
    axes[0, 1].set_title('箱线图')
    axes[0, 1].set_ylabel('数值')
    axes[0, 1].grid(alpha=0.3, axis='y')

    # 3. KDE图
    sns.kdeplot(data, ax=axes[1, 0], fill=True, alpha=0.5)
    axes[1, 0].set_title('核密度估计图')
    axes[1, 0].set_xlabel('数值')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].grid(alpha=0.3)

    # 4. 小提琴图
    sns.violinplot(y=data, ax=axes[1, 1], inner="quartile")
    axes[1, 1].set_title('小提琴图')
    axes[1, 1].set_ylabel('数值')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # 打印统计信息
    print(f"数据统计摘要：")
    print(f"样本数: {len(data)}")
    print(f"均值: {np.mean(data):.4f}")
    print(f"标准差: {np.std(data):.4f}")
    print(f"最小值: {np.min(data):.4f}")
    print(f"25%分位数: {np.percentile(data, 25):.4f}")
    print(f"中位数: {np.median(data):.4f}")
    print(f"75%分位数: {np.percentile(data, 75):.4f}")
    print(f"最大值: {np.max(data):.4f}")


# 定义函数：为每组数据构建经验CDF映射函数
def build_ecdf_mapper(returns: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    构建经验累积分布函数（ECDF）映射器。
    参数:
        returns: 一维数组，历史收益率数据
    返回:
        mapper: 函数，输入新收益率数组，输出映射到[0,1]的分数
    """
    # 步骤1: 对原始收益率排序
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)

    # 步骤2: 计算ECDF值（均匀阶梯，从1/n到1）
    ecdf_values = np.arange(1, n + 1) / n

    def mapper(new_returns: np.ndarray) -> np.ndarray:
        """
        映射函数：将新收益率映射到[0,1]区间
        参数:
            new_returns: 一维数组，新收益率数据
        返回:
            scores: 一维数组，映射后的分数
        """
        # 使用插值：对于每个新收益率，在排序收益率中找到其位置，线性插值得到ECDF值
        # np.interp用于一维线性插值，要求x坐标（sorted_returns）必须递增
        scores = np.interp(new_returns, sorted_returns, ecdf_values)
        return scores

    return mapper

# preload()
# 2. 从 .pkl 文件中读取字典
def preprocess_ECDF_dprofit():
    with open('1d-m-data-120d.pkl', 'rb') as file: # 注意是 'rb' 模式
        date_list = pickle.load(file)
    print("读取的字典内容：", date_list.keys())

    start_year = 2010
    end_year = 2025
    months = 12
    days = 31

    target_profit = []
    data_num = 0
    origin_num = 0

    scaler = MinMaxScaler(feature_range=(0, 1))  # 或者 StandardScaler()

    year_index = {}
    features_data = []

    _2s3h_data = []

    for y in range(start_year, end_year + 1):
        year_index[str(y)]=data_num
        for m in range(months):
            for d in range(days):
                if date_list[str(y)][str(m + 1)][str(d + 1)]['list']:

                    date_list[str(y)][str(m + 1)][str(d + 1)]['list'].sort(key=lambda x: x[1]['d_profit'], reverse=True)
                    date_list[str(y)][str(m + 1)][str(d + 1)]['num'] = len(
                        date_list[str(y)][str(m + 1)][str(d + 1)]['list'])

                    top_num = 0
                    profit_conf = 0.01
                    for x in date_list[str(y)][str(m + 1)][str(d + 1)]['list']:
                        origin_num += 1
                        # print(x[1]['data'].columns)
                        print(x)
                        features_df = x[1]['data'][
                            ['开盘', '最高', '最低', '收盘', '成交量', '中轨', '上轨', '下轨', 'DIFF', 'DEA', 'MACD']]

                        # 2. 调用上面的特征工程函数
                        features_normalized = engineer_features_f11(features_df)

                        features_data.append(features_normalized)


                        # ###################### 绝对值处理方法 ################################
                        # data_np = x[1]['data'].to_numpy()
                        # # print(data_np.shape)
                        # data_np = data_np[:, 2:]
                        #
                        # nan_count = np.isnan(data_np).sum()
                        #
                        # if nan_count > 0:
                        #     print('缺失值：', nan_count)
                        #     break
                        #
                        # features_normalized = scaler.fit_transform(data_np)
                        # features_data.append(features_normalized)

                        # print(features_normalized.shape)
                        # print(features_normalized)
                        # print(data_np.shape)
                        # print(data_np)
                        # print(x[1]['info'])

                        if top_num >= 10000:
                            break
                        profit = x[1]['d_profit']

                        target_profit.append(x[1]['d_profit'])
                        data_num += 1
                        top_num += 1

                        # if profit > profit_conf:
                        #     target_profit.append(x[1]['d_profit'])
                        #     data_num += 1
                        #     top_num += 1
                        print(data_num)

                else:
                    date_list[str(y)][str(m + 1)].pop(str(d + 1))

    mapper = build_ecdf_mapper(target_profit)
    print(origin_num, data_num)
    target_mapper = mapper(target_profit)
    plot_data_distribution(target_mapper)
    features_data = np.array(features_data)
    print(features_data.shape)
    print(target_mapper.shape)
    print(year_index)
    data_pkl = {"features_data": features_data, "target_mapper": target_mapper, "year_index": year_index}

    for i in range(100):
        print(target_profit[i], target_mapper[i])

    with open('1d-m-data-120d-preprocess-8f.pkl', 'wb') as file: # 注意是 'wb' 模式
        pickle.dump(data_pkl, file)
    print("字典已保存！")

def preprocess_class(buffer_zone=False, buffer_area=(0.00, 0.03), preprocess_type='2s3h', feature_length=120):
    """
    preprocess_type: 2s3h/2s3v/2s3e 收益率指标
    feature_length: 30/60/120 特征切片长度
    """
    with open('dataset/0d-m-data-150d.pkl', 'rb') as file: # 注意是 'rb' 模式
        date_list = pickle.load(file)
    print("读取的字典内容：", date_list.keys())

    start_year = 2010
    end_year = 2025
    months = 12
    days = 31

    target_profit = []
    data_num = 0
    origin_num = 0

    year_index = {}
    year_index_list = {}
    features_data = []

    _2s3h_data = []
    label_list = []
    info_list = []


    for y in range(start_year, end_year + 1):
        if buffer_zone:
            year_index[str(y)] = data_num
        else:
            year_index[str(y)] = origin_num
        year_index_list[str(y)] = {}

        print(year_index_list)

        for m in range(months):
            for d in range(days):
                if date_list[str(y)][str(m + 1)][str(d + 1)]['list']:

                    day_index = str(y) + '-' + str(m + 1) + '-' + str(d + 1)
                    if not buffer_zone:
                        year_index_list[str(y)][day_index] = origin_num

                    date_list[str(y)][str(m + 1)][str(d + 1)]['list'].sort(key=lambda x: x[1]['d_profit'], reverse=True)
                    date_list[str(y)][str(m + 1)][str(d + 1)]['num'] = len(
                        date_list[str(y)][str(m + 1)][str(d + 1)]['list'])

                    top_num = 0
                    profit_conf = 0.03
                    buffer_zone_high = buffer_area[1]
                    buffer_zone_low = buffer_area[0]

                    for x in date_list[str(y)][str(m + 1)][str(d + 1)]['list']:
                        origin_num += 1
                        # print(x[1]['data'].columns)
                        # print(x)
                        _2s = x[1]['day2-info']['start']
                        _3s = x[1]['day3-info']['start']
                        _3h = x[1]['day3-info']['max']
                        _3e = x[1]['day3-info']['end']
                        _3l = x[1]['day3-info']['min']
                        _3v = (_3e + _3l + _3h) / 3
                        preprocess_profit = (_3h - _2s) / _2s

                        info_list.append({'code':x[0], '2s':_2s, '3s':_3s, '3h':_3h, '3v':_3v,
                                          '3e':_3e, '3l':_3l, 'date':day_index})

                        if preprocess_type == '2s3h':
                            preprocess_profit = (_3h - _2s) / _2s
                        elif preprocess_type == '2s3v':
                            preprocess_profit = (_3v - _2s) / _2s
                        elif preprocess_type == '2s3e':
                            preprocess_profit = (_3e - _2s) / _2s

                        # 隔离区判别
                        if buffer_zone:
                            if preprocess_profit > buffer_zone_high:
                                _2s3h_data.append(preprocess_profit)
                                label_list.append(1)
                                data_num += 1
                            elif preprocess_profit < buffer_zone_low:
                                label_list.append(0)
                                data_num += 1
                            else:
                                continue
                        else:
                            if preprocess_profit > profit_conf:
                                _2s3h_data.append(preprocess_profit)
                                label_list.append(1)
                                data_num += 1
                            else:
                                label_list.append(0)


                        features_df = x[1]['data'][
                            ['开盘', '最高', '最低', '收盘', '成交量', '中轨', '上轨', '下轨', 'DIFF', 'DEA', 'MACD']]

                        # 2. 调用上面的特征工程函数
                        features_normalized = engineer_features_f14(features_df)
                        features_normalized = features_normalized[-feature_length:,:]

                        features_data.append(features_normalized)

                        profit = x[1]['d_profit']

                        target_profit.append(profit)

                        top_num += 1

                        print(data_num)

                else:
                    date_list[str(y)][str(m + 1)].pop(str(d + 1))


    features_data = np.array(features_data)
    label_list = np.array(label_list)
    print(features_data.shape)
    print(label_list.shape)
    print(len(info_list))
    print(year_index)
    print(origin_num)
    print(np.mean(_2s3h_data))
    print(np.mean(target_profit))
    print(len(_2s3h_data))
    print(data_num)
    data_pkl = {"features_data": features_data, "label_list": label_list, "year_index": year_index,
                "year_index_list": year_index_list, "info_list": info_list}

    with open(f'dataset/0d-m-data-{feature_length}d-preprocess-14f-{preprocess_type}-class-003-TEST.pkl', 'wb') as file: # 注意是 'wb' 模式
        pickle.dump(data_pkl, file)
    print("字典已保存！")

def create_data_loaders(dataset_preprocess_name='0d-m-data-120d-preprocess.pkl', batch_size=32, data_type='class',
                        feature_length=120):
    """创建数据加载器"""
    with open(dataset_preprocess_name, 'rb') as file: # 注意是 'rb' 模式
        data_pkl = pickle.load(file)
    print("读取的字典内容：", data_pkl.keys())

    features_data = data_pkl['features_data']

    year_index = data_pkl['year_index']
    if data_type == 'class':
        features_label = data_pkl['label_list']
        label_rate = sum(features_label) / len(features_label)
        print("正样本所占比率：", label_rate)
    else:
        features_label = data_pkl['target_mapper']

    train_data = features_data[:year_index['2022'], -feature_length:,:]
    val_data = features_data[year_index['2022']:year_index['2024'], -feature_length:, :]
    test_data = features_data[year_index['2024']:, -feature_length:, :]

    train_label = features_label[:year_index['2022'],]
    val_label = features_label[year_index['2022']:year_index['2024'],]
    test_label = features_label[year_index['2024']:,]

    print(train_label.shape, val_label.shape, test_label.shape)
    print(train_data.shape, val_data.shape, test_data.shape)

    X_train = torch.FloatTensor(train_data)
    y_train = torch.FloatTensor(train_label).unsqueeze(1)  # 形状 (n, 1)
    X_val = torch.FloatTensor(val_data)
    y_val = torch.FloatTensor(val_label).unsqueeze(1)
    X_test = torch.FloatTensor(test_data)
    y_test = torch.FloatTensor(test_label).unsqueeze(1)

    # 方法2：统计NaN值的总数
    nan_count_1 = torch.isnan(X_train).sum()
    nan_count_2 = torch.isnan(X_val).sum()
    nan_count_3 = torch.isnan(X_test).sum()

    print(f"张量1中NaN值的个数: {nan_count_1.item()}")
    print(f"张量2中NaN值的个数: {nan_count_2.item()}")
    print(f"张量3中NaN值的个数: {nan_count_3.item()}")

    print(X_train.shape, X_val.shape, X_test.shape,
          y_train.shape, y_val.shape, y_test.shape)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_test, y_test


if __name__ == '__main__':
    # preload(length=150)
    # preprocess_class(buffer_zone=True, buffer_area=(0.01, 0.03), preprocess_type='2s3e', feature_length=120)
    preprocess_class(buffer_zone=False, preprocess_type='2s3e', feature_length=120)

