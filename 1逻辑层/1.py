# PyTorch中AI个体(神经网络模型)的建立详解

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. 定义神经网络模型类
# 继承nn.Module是创建PyTorch模型的基础
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层
        self.dropout = nn.Dropout(0.2)  #  dropout层用于防止过拟合

    def forward(self, x):
        # 定义前向传播
        x = F.relu(self.fc1(x))  # 输入层 -> ReLU激活
        x = self.dropout(x)  # 应用dropout
        x = F.relu(self.fc2(x))  # 隐藏层 -> ReLU激活
        x = self.fc3(x)  # 输出层
        return x

# 2. 数据准备
# 生成一些示例数据
def prepare_data():
    # 生成随机输入和标签
    X = torch.randn(1000, 10)  # 1000个样本，每个样本10个特征
    y = torch.randint(0, 3, (1000,))  # 1000个标签，3个类别
    
    # 划分训练集和测试集
    train_size = 800
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# 3. 模型训练
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # 设置模型为训练模式
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # 计算 epoch 损失
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 4. 模型评估
def evaluate_model(model, test_loader, criterion):
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item() * inputs.size(0)
            
            # 计算预测正确的数量
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    # 计算平均损失和准确率
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
    return test_loss, accuracy

# 5. 主函数：整合所有步骤
def main():
    # 超参数设置
    input_size = 10
    hidden_size = 64
    output_size = 3
    learning_rate = 0.001
    num_epochs = 10
    
    # 创建模型
    model = SimpleNN(input_size, hidden_size, output_size)
    print(model)  # 打印模型结构
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 准备数据
    train_loader, test_loader = prepare_data()
    
    # 训练模型
    print("开始训练模型...")
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # 评估模型
    print("评估模型性能...")
    evaluate_model(model, test_loader, criterion)
    
    # 保存模型
    torch.save(model.state_dict(), 'simple_nn_model.pth')
    print("模型已保存到 'simple_nn_model.pth'")

if __name__ == '__main__':
    main()

