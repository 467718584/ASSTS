import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn
import torch.nn.functional as F
import torch
from Class_score.model.LSTM import TimeSeriesScorerLSTM
from Class_score.model.CNN import TimeSeriesScorer1DCNN
from data_preload import create_data_loaders

class TimeSeriesTrainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()  # 均方误差损失
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.device = device

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for batch_X, batch_y in self.train_loader:
            self.optimizer.zero_grad()

            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def train(self, epochs=100, patience=50):
        early_stop_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            # 早停机制
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 加载最佳模型
        torch.save(self.model.state_dict(), 'best_model.pth')
        self.model.load_state_dict(torch.load('best_model.pth'))


def main_training_pipeline():
    """完整的训练流程"""
    # 1. 数据预处理
    # X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(features, scores)

    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 创建数据加载器
    batch_size = 256
    dataset_name = '1d-m-data-120d-preprocess.pkl'
    train_loader, val_loader, test_loader , X_test, y_test = create_data_loaders(dataset_name, batch_size)

    # 3. 初始化模型（选择1D CNN或LSTM）
    model = TimeSeriesScorer1DCNN(input_channels=11, time_steps=121).to(device)
    # model = TimeSeriesScorer1DCNN(input_channels=11, time_steps=121)
    # 或者: model = TimeSeriesScorerLSTM(input_size=11)

    # 4. 训练模型
    trainer = TimeSeriesTrainer(model, train_loader, val_loader, lr=0.001, device=device)
    trainer.train(epochs=100, patience=50)

    # 5. 测试集评估
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            test_predictions = model(batch_X)
            test_loss = nn.MSELoss()(test_predictions, batch_y)

            # 计算评估指标
            test_rmse = torch.sqrt(test_loss).item()
            test_r2 = r2_score(y_test.numpy(), test_predictions.cpu().numpy())

            print(f"Test Results - RMSE: {test_rmse:.4f}, R² Score: {test_r2:.4f}")

    return model, trainer

# 使用示例
# 假设您的数据格式如下：
# features = np.random.randn(10000, 120, 11)  # 示例数据
# scores = np.random.rand(10000)  # 示例评分
if __name__ == '__main__':
    model, trainer = main_training_pipeline()
