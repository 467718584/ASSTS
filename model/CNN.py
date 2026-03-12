import torch.nn as nn
import torch.nn.functional as F
import torch


class TimeSeriesScorer1DCNN(nn.Module):
    def __init__(self, input_channels=11, time_steps=120, dropout_rate=0.3):
        super(TimeSeriesScorer1DCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # 第一卷积块
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate),

            # 第二卷积块
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate),

            # 第三卷积块
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate)
        )

        # 计算卷积层输出尺寸
        conv_output_size = self._get_conv_output_size(time_steps, input_channels)

        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            # nn.Sigmoid()  # 输出在0-1之间
        )

    def _get_conv_output_size(self, time_steps, input_channels):
        """计算卷积层输出尺寸"""
        with torch.no_grad():
            x = torch.zeros(1, input_channels, time_steps)
            x = self.conv_layers(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        # 输入x形状: (batch_size, 120, 11)
        # 转换为Conv1d期望的形状: (batch_size, channels, length)
        x = x.transpose(1, 2)  # 形状变为 (batch_size, 11, 120)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x