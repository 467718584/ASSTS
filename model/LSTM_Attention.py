import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesScorerLSTM_Attn(nn.Module):
    def __init__(self, input_channels=11, time_steps=120, hidden_dim=64, num_layers=2, dropout_rate=0.3):
        super(TimeSeriesScorerLSTM_Attn, self).__init__()

        # 1. Bi-LSTM 层: 提取时间序列特征
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # 双向
            dropout=dropout_rate
        )

        # 2. Attention 层: 计算每个时间步的权重
        # 维度 * 2 是因为双向 LSTM
        self.attention_linear = nn.Linear(hidden_dim * 2, 1)

        # 3. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)  # 输出 Logits
        )

    def forward(self, x):
        # x: (batch, time_steps, channels)

        # LSTM 输出: (batch, time_steps, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # Attention 机制
        # 计算得分: (batch, time_steps, 1)
        attn_weights = self.attention_linear(lstm_out)
        # Softmax 归一化: (batch, time_steps, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # 加权求和: (batch, hidden_dim * 2)
        # 这步操作让模型自动“关注”最重要的时间点
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        # 分类
        logits = self.classifier(context_vector)
        return logits