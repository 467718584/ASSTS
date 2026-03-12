import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch.nn as nn
import torch
import numpy as np
from Class_score.model.CNN import TimeSeriesScorer1DCNN
from Class_score.model.LSTM import TimeSeriesScorerLSTM
from Class_score.model.Transformer import TimeSeriesTransformer
from data_preload import create_data_loaders


class TimeSeriesTrainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.device = device

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        count = 0

        for batch_X, batch_y in self.train_loader:
            self.optimizer.zero_grad()

            # --- 安全检查：防止脏数据导致 NaN ---
            if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                # print("Warning: NaN/Inf found in input batch. Replacing with 0.")
                batch_X = torch.nan_to_num(batch_X, nan=0.0, posinf=1.0, neginf=0.0)

            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            if torch.isnan(loss):
                print("Error: Loss is NaN inside training loop!")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            count += 1

        return epoch_loss / count if count > 0 else float('inf')

    def validate(self):
        self.model.eval()
        val_loss = 0
        count = 0

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = torch.nan_to_num(batch_X, nan=0.0)  # 验证集也要保护
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
                count += 1

        return val_loss / count if count > 0 else float('inf')

    def train(self, epochs=100, patience=50):
        early_stop_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 5 == 0:  # 稍微频繁一点打印
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 加载最佳模型
        print("Loading best model for testing...")
        self.model.load_state_dict(torch.load('best_model.pth'))


def main_training_pipeline():
    """完整的训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 创建数据加载器
    batch_size = 512
    dataset_name = '0d-m-data-120d-preprocess-8f.pkl'  # 确保这里读取的是包含120步数据的文件
    # 注意：如果你的pkl还没重新生成，这里读出来的可能还是121步
    train_loader, val_loader, test_loader, X_test, y_test = create_data_loaders(dataset_name, batch_size)

    # 获取输入维度
    sample_X, _ = next(iter(train_loader))
    time_steps = sample_X.shape[1]  # 自动获取 120 或 121
    input_channels = sample_X.shape[2]
    print(f"Detected Input Shape: Time Steps={time_steps}, Channels={input_channels}")

    # 2. 初始化模型
    # model = TimeSeriesScorer1DCNN(input_channels=input_channels, time_steps=time_steps).to(device)
    # model = TimeSeriesScorerLSTM(input_size=input_channels, hidden_size=128, num_layers=2).to(device)
    model = TimeSeriesTransformer(input_size=input_channels, d_model=64, nhead=4, num_layers=3).to(device)

    # 3. 训练模型
    trainer = TimeSeriesTrainer(model, train_loader, val_loader, lr=0.001, device=device)
    trainer.train(epochs=100, patience=50)

    # 4. 测试集评估 (修复了这里)
    model.eval()

    # 用来收集所有的预测值和真实值
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = torch.nan_to_num(batch_X, nan=0.0)
            batch_X = batch_X.to(device)
            # batch_y 不需要 to(device) 因为最后要转回 cpu 存列表，或者统一处理

            test_predictions = model(batch_X)

            # 将预测结果移回 CPU 并加入列表
            all_preds.append(test_predictions.cpu())
            all_targets.append(batch_y)  # batch_y 本身在cpu上 (dataset中是cpu tensor)

    # 拼接所有批次的结果
    if len(all_preds) > 0:
        y_pred_tensor = torch.cat(all_preds).numpy()
        y_true_tensor = torch.cat(all_targets).numpy()

        # 计算评估指标
        mse_loss = mean_squared_error(y_true_tensor, y_pred_tensor)
        test_rmse = np.sqrt(mse_loss)
        test_r2 = r2_score(y_true_tensor, y_pred_tensor)
        test_mae = mean_absolute_error(y_true_tensor, y_pred_tensor)

        print(f"\n===== Test Results =====")
        print(f"RMSE:     {test_rmse:.4f}")
        print(f"MAE:      {test_mae:.4f}")
        print(f"R² Score: {test_r2:.4f}")

        # 简单查看前10个预测对比
        print("\nSample predictions (True vs Pred):")
        for i in range(10):
            print(f"{y_true_tensor[i][0]:.4f} vs {y_pred_tensor[i][0]:.4f}")
    else:
        print("Error: Test loader is empty.")

    return model, trainer


if __name__ == '__main__':
    model, trainer = main_training_pipeline()