from xml.sax.handler import all_properties

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, '/home/zzy/project/ASSTS/ASSTS-stock_class')
from model.LSTM_Attention import TimeSeriesScorerLSTM_Attn
import pickle
import json
import pandas as pd

def json_to_excel_detailed(json_data, output_excel_path):
    # 1. 加载 JSON 数据
    # with open(json_file_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    data = json_data

    # 2. 准备“每日汇总”数据 (Daily Summary)
    # 包含 Fund_1, Fund_2 观察资金交替情况
    daily_list = []
    all_trades = []

    for day_data in data:
        # 提取每日概要
        daily_info = {
            "交易日期": day_data.get('trade_day'),
            "资金池1 (Fund_1)": day_data.get('Fund_1'),
            "资金池2 (Fund_2)": day_data.get('Fund_2'),
            "当日收益率": day_data.get('all_profit'),
            "当日收益额": day_data.get('all_profit_charge'),
            "总手续费率": day_data.get('all_handing_rate'),
            "总手续费额": day_data.get('all_handing_charge')
        }
        daily_list.append(daily_info)

        # 3. 准备“交易明细”数据 (Detailed Trades)
        # 将每日的 day_trade_list 展开
        for trade in day_data.get('day_trade_list', []):
            trade_info = {
                "交易日期": day_data.get('trade_day'),
                "股票代码": trade.get('code'),
                "买入价格": trade.get('buy_value'),
                "卖出价格": trade.get('sell_value'),
                "个股收益率": trade.get('profit'),
                "个股收益额": trade.get('profit_charge'),
                "个股手续费": trade.get('handing_charge')
            }
            all_trades.append(trade_info)

    # 转化为 DataFrame
    df_daily = pd.DataFrame(daily_list)
    df_trades = pd.DataFrame(all_trades)

    # 4. 写入 Excel
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        # Sheet 1: 每日资金与收益汇总
        df_daily.to_excel(writer, sheet_name='每日资金收益汇总', index=False)
        # Sheet 2: 逐笔交易清单
        df_trades.to_excel(writer, sheet_name='逐笔交易明细', index=False)

    print(f"转换完成！Excel 文件已保存至: {output_excel_path}")

# 使用示例


class StockPredictor:
    def __init__(self, model_path, input_channels=14, time_steps=30, device=None):
        """
        初始化预测器
        :param model_path: 训练好的模型权重路径 (例如 'best_model.pth')
        :param input_channels: 特征数量 (根据你的实验，目前应为 14)
        :param time_steps: 时间步长 (30, 60 或 120)
        :param device: 推理设备
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型结构 (默认使用表现最好的 LSTM+Attn)
        self.model = TimeSeriesScorerLSTM_Attn(
            input_channels=input_channels,
            time_steps=time_steps
        ).to(self.device)

        # 加载权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功加载模型权重: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")

        self.model.eval()  # 切换至评估模式

    def predict(self, features_np):
        """
        进行预测
        :param features_np: 输入特征，形状为 (Batch, TimeSteps, Channels) 或 (TimeSteps, Channels)
        :return: 预测类别 (0/1), 置信度 (0.0~1.0)
        """
        # 1. 维度处理：如果是单条数据 (TimeSteps, Channels)，增加 Batch 维度
        if len(features_np.shape) == 2:
            features_np = np.expand_dims(features_np, axis=0)

        # 2. 数据转换与 NaN 处理
        features_tensor = torch.from_numpy(features_np).float()
        features_tensor = torch.nan_to_num(features_tensor, nan=0.0).to(self.device)

        with torch.no_grad():
            # 3. 推理得到 Logits
            logits = self.model(features_tensor)

            # 4. 计算置信度 (Sigmoid 转换)
            # 在二分类中，Sigmoid 输出的值即为预测为类别 1 的概率
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

            # 5. 判定类别 (以 0.5 为阈值)
            classes = (probabilities > 0.5).astype(int)

        return classes, probabilities


def predict_result(dataset_path='', MODEL_WEIGHTS='', feature_length=120, conf=0.5):
    DATASET_test = dataset_path
    with open(DATASET_test, 'rb') as file: # 注意是 'rb' 模式
        data_pkl = pickle.load(file)
    print("读取的字典内容：", data_pkl.keys())
    print(data_pkl['year_index_list'])

    # print(data_pkl['info_list'])
    info_list = data_pkl['info_list']
    feature_data = data_pkl['features_data']

    last_index = 0
    last_day = ''
    info_num = 0
    feature_length = feature_length
    day_predict_dict = {}
    day_predict_result = {}
    features = 14
    conf = conf
    top_num = 10

    predictor = StockPredictor(MODEL_WEIGHTS, input_channels=features, time_steps=feature_length)
    #
    # # 模拟回测中从特征工程模块获取的输入数据 (BatchSize=10, TimeSteps=30, Channels=14)
    # mock_input = np.random.randn(10, STEPS, CHANNELS)
    #
    # # 获取预测结果
    ending_year = '2023'
    test_year_list = ['2024', '2025']

    if ending_year != '2010':
        print(data_pkl['year_index_list'][ending_year])
        last_index_dict = data_pkl['year_index_list'][ending_year]
        last_index = list(last_index_dict.values())[-1]

    for year_index in test_year_list:
    # for year_index in data_pkl['year_index_list']:
        for day_index in data_pkl['year_index_list'][year_index]:
            # print(day_index, data_pkl['year_index_list'][year_index][day_index])
            day_index_num = data_pkl['year_index_list'][year_index][day_index]
            if day_index_num == 0:
                pass
            else:
                day_info_list = info_list[last_index:day_index_num]
                info_num += len(day_info_list)
                day_feature = feature_data[last_index:day_index_num,-feature_length:,:]

                print(day_feature.shape)

                day_predict_dict[day_index] = {'info': day_info_list, 'feature': day_feature}

                day_predict_result[day_index] = []
                classes, confidence = predictor.predict(day_feature)
                sorted_indices = [index for index, _ in sorted(enumerate(confidence), key=lambda x: x[1], reverse=True)]
                sorted_indices = sorted_indices[:top_num]
                print(sorted_indices)

                for i in sorted_indices:
                    if confidence[i] >= conf:
                        print(
                            f"样本 {i}: 预测类别={'看涨' if classes[i] == 1 else '看淡'}, "
                            f"置信度={confidence[i]:.4f}, {day_info_list[i]}")
                        day_predict_result[day_index].append({'confidence': confidence[i],'day_info': day_info_list[i]})

                # for i in range(len(classes)):
                #     print(
                #         f"样本 {i + 1}: 预测类别={'看涨' if classes[i] == 1 else '看淡'}, "
                #         f"置信度={confidence[i]:.4f}, {day_info_list[i]}")

                # print(day_index_num, info_num)
                # print(last_day)
                # print(day_info_list)
                # print(day_feature.shape)
                last_index = day_index_num

            last_day = (day_index, data_pkl['year_index_list'][year_index][day_index])

    if info_num != len(info_list):
        day_info_list = info_list[last_index:]
        info_num += len(day_info_list)
        day_feature = feature_data[last_index:, -feature_length:, :]
        day_predict_dict[last_day[0]] = {'info': day_info_list, 'feature': day_feature}

        day_predict_result[last_day[0]] = []
        classes, confidence = predictor.predict(day_feature)
        sorted_indices = [index for index, _ in sorted(enumerate(confidence), key=lambda x: x[1], reverse=True)]
        sorted_indices = sorted_indices[:5]
        print(sorted_indices)

        for i in sorted_indices:
            if confidence[i] >= conf:
                print(
                    f"样本 {i}: 预测类别={'看涨' if classes[i] == 1 else '看淡'}, "
                    f"置信度={confidence[i]:.4f}, {day_info_list[i]}")
                day_predict_result[last_day[0]].append({'confidence': confidence[i], 'day_info': day_info_list[i]})


        # print(day_index_num, info_num)
        # print(last_day)
        # print(day_info_list)

    print(len(info_list), info_num)
    print(len(day_predict_dict), day_predict_dict.keys())
    # print(day_predict_dict)
    return day_predict_dict, day_predict_result


# === 回测环境下的调用示例 ===
if __name__ == '__main__':
    # 配置参数
    MODEL_WEIGHTS = 'model_file/0d_120d_14f_2s3h_BCE_0005-003_AUC_05843.pth'
    DATASET_test = "dataset/0d-m-data-120d-preprocess-14f-2s3e-class-003-TEST.pkl"
    f_length = 120
    conf = 0.5
    test_dataset, test_predict_results = predict_result(DATASET_test, MODEL_WEIGHTS, f_length, conf)

    # print(f"test_dataset: {test_dataset}")
    # print(f"test_predict_results: {test_predict_results}")

    Fund_POOL = [50000, 50000]
    trading_day = 0
    trade_list = []
    for day_index in test_predict_results:

        if len(test_predict_results[day_index]) == 0:
            continue

        trade_info = {"trade_day": day_index, "Fund_1": Fund_POOL[0], "Fund_2": Fund_POOL[1]}
        Fund_ID = int(trading_day % 2)
        trading_day += 1

        fund = Fund_POOL[Fund_ID]
        fund_one = fund/len(test_predict_results[day_index])
        day_trade_list = []
        new_fund = 0
        handing_rate = 0.00015 + 0.00015 + 0.0005  # 买入佣金 + 卖出佣金 + 印花税
        # handing_rate = 0

        profit_list = []

        for day_info_dict in test_predict_results[day_index]:
            day_info = day_info_dict['day_info']
            code = day_info['code']
            _2s = day_info['2s']
            _3s = day_info['3s']
            _3h = day_info['3h']
            _3e = day_info['3e']
            _3v = day_info['3v']
            # profit = (_3e - _2s) / _2s
            # p = (_3h - _2s) / _2s

            buy_value = _2s
            sell_value = _3v
            p = (sell_value - buy_value) / buy_value

            new_fund += (1 + p - handing_rate) * fund_one
            profit_list.append(p)

            day_trade_list.append({"code":code,
                                   "buy_value": buy_value,
                                   "sell_value": sell_value,
                                   "profit":p,
                                   "profit_charge":fund_one*p,
                                   "handing_charge":fund_one*handing_rate})


        all_profit = float(np.mean(profit_list))
        trade_info.update({'all_profit': all_profit,
                           'all_profit_charge': fund*all_profit,
                           'all_handing_rate': handing_rate,
                           'all_handing_charge': fund*handing_rate,
                           'day_trade_list': day_trade_list})
        trade_list.append(trade_info)

        print(profit_list)

        Fund_POOL[Fund_ID] = new_fund

        print(Fund_POOL)
        print(trade_info)
    print(sum(Fund_POOL))

    output_xlsx = '回测系统交易分析报告.xlsx'
    json_to_excel_detailed(trade_list, output_xlsx)

    json_data = json.dumps(trade_list, indent=4)
    with open('0d-m-data-120d-14f-2s3e_model-2s3e_profit.json', 'w') as file:
        file.write(json_data)













