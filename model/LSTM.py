import torch
import torch.nn as nn


class TimeSeriesScorerLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, num_layers=2, dropout_rate=0.2):
        super(TimeSeriesScorerLSTM, self).__init__()

        # 使用双向 LSTM，增强信息提取能力
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True  # 关键：双向
        )

        # 引入 BatchNorm 加速收敛
        self.bn = nn.BatchNorm1d(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # 双向所以是 hidden_size * 2
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            # nn.Sigmoid()  # 保证输出在 0-1 之间
        )

    def forward(self, x):
        # x shape: (batch_size, 121, 11)

        # LSTM 输出: output, (h_n, c_n)
        # output shape: (batch, seq_len, num_directions * hidden_size)
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出作为特征
        # 注意：双向LSTM通常取最后一步的正向和第一步的反向，或者直接取最后一步的拼接
        last_output = lstm_out[:, -1, :]

        last_output = self.bn(last_output)
        output = self.classifier(last_output)
        return output