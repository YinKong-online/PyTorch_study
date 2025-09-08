# 空洞CNN算法实现
# 项目：语义分割高精度实现
# 目标：使用空洞卷积技术构建高效的语义分割模型，在保持高空间分辨率的同时扩大感受野，提高分割精度
# 步骤：
# 1. 导入必要的库和模块
# 2. 定义空洞CNN网络结构
# 3. 设置数据集加载和预处理
# 4. 定义训练和评估函数
# 5. 实现主函数进行训练和测试
# 结构：
# - 编码器：使用VGG16作为基础网络，替换部分卷积为空洞卷积
# - 解码器：使用上采样和跳跃连接恢复空间分辨率
# - 分类层：输出像素级分类结果
# 损失函数：交叉熵损失函数结合Dice系数，提高边界分割精度

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "NSimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义空洞卷积模块
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding * dilation,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 定义基于空洞卷积的语义分割网络
class DilatedCNN(nn.Module):
    def __init__(self, num_classes=21):  # PASCAL VOC数据集有21个类别
        super(DilatedCNN, self).__init__()
        
        # 编码器部分 - 使用VGG16的前几层结构，但替换为空洞卷积
        # 第一层 - 不使用空洞卷积
        self.encoder1 = nn.Sequential(
            DilatedConvBlock(3, 64, kernel_size=3, padding=1, dilation=1),
            DilatedConvBlock(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二层 - 不使用空洞卷积
        self.encoder2 = nn.Sequential(
            DilatedConvBlock(64, 128, kernel_size=3, padding=1, dilation=1),
            DilatedConvBlock(128, 128, kernel_size=3, padding=1, dilation=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第三层 - 不使用空洞卷积
        self.encoder3 = nn.Sequential(
            DilatedConvBlock(128, 256, kernel_size=3, padding=1, dilation=1),
            DilatedConvBlock(256, 256, kernel_size=3, padding=1, dilation=1),
            DilatedConvBlock(256, 256, kernel_size=3, padding=1, dilation=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第四层 - 使用空洞卷积，dilation=2
        self.encoder4 = nn.Sequential(
            DilatedConvBlock(256, 512, kernel_size=3, padding=1, dilation=2),
            DilatedConvBlock(512, 512, kernel_size=3, padding=1, dilation=2),
            DilatedConvBlock(512, 512, kernel_size=3, padding=1, dilation=2),
            # 不使用池化，保持空间分辨率
        )
        
        # 第五层 - 使用空洞卷积，dilation=4
        self.encoder5 = nn.Sequential(
            DilatedConvBlock(512, 512, kernel_size=3, padding=1, dilation=4),
            DilatedConvBlock(512, 512, kernel_size=3, padding=1, dilation=4),
            DilatedConvBlock(512, 512, kernel_size=3, padding=1, dilation=4),
            # 不使用池化，保持空间分辨率
        )
        
        # 分类层
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # 上采样层 - 双线性插值上采样到原始图像大小
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # 编码器前向传播
        x1 = self.encoder1(x)  # (batch_size, 64, H/2, W/2)
        x2 = self.encoder2(x1)  # (batch_size, 128, H/4, W/4)
        x3 = self.encoder3(x2)  # (batch_size, 256, H/8, W/8)
        x4 = self.encoder4(x3)  # (batch_size, 512, H/8, W/8) - 使用空洞卷积，保持分辨率
        x5 = self.encoder5(x4)  # (batch_size, 512, H/8, W/8) - 使用空洞卷积，保持分辨率
        
        # 分类
        x = self.classifier(x5)  # (batch_size, num_classes, H/8, W/8)
        
        # 上采样到原始图像大小
        x = self.upsample(x)  # (batch_size, num_classes, H, W)
        
        return x

# 定义Dice损失函数，提高边界分割精度
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, preds, targets):
        # 将预测转换为one-hot编码
        preds = nn.functional.softmax(preds, dim=1)
        
        # 计算交并比
        intersection = torch.sum(preds * targets, dim=(2, 3))
        union = torch.sum(preds + targets, dim=(2, 3))
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 计算损失（1 - Dice系数的平均值）
        loss = 1 - dice.mean()
        
        return loss

# 定义总损失函数：交叉熵损失 + Dice损失
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, preds, targets):
        # 交叉熵损失需要的是类别索引，而不是one-hot编码
        ce_loss_val = self.ce_loss(preds, targets.argmax(dim=1))
        dice_loss_val = self.dice_loss(preds, targets)
        
        # 加权组合损失
        total_loss = self.ce_weight * ce_loss_val + self.dice_weight * dice_loss_val
        
        return total_loss

# 定义数据预处理和加载函数
def get_dataloaders(batch_size=4, image_size=224):
    """创建模拟数据集用于测试"""
    # 创建随机图像数据 (batch_size * 10, 3, image_size, image_size)
    num_samples = 100
    train_images = torch.randn(num_samples, 3, image_size, image_size)
    val_images = torch.randn(num_samples // 2, 3, image_size, image_size)
    
    # 创建随机标签 (batch_size * 10, num_classes, image_size, image_size)
    num_classes = 21
    train_targets = torch.randint(0, num_classes, (num_samples, 1, image_size, image_size)).float()
    val_targets = torch.randint(0, num_classes, (num_samples // 2, 1, image_size, image_size)).float()
    
    # 转换为one-hot编码
    train_targets_onehot = torch.zeros(num_samples, num_classes, image_size, image_size)
    val_targets_onehot = torch.zeros(num_samples // 2, num_classes, image_size, image_size)
    
    for i in range(num_samples):
        for c in range(num_classes):
            train_targets_onehot[i, c] = (train_targets[i, 0] == c).float()
    
    for i in range(num_samples // 2):
        for c in range(num_classes):
            val_targets_onehot[i, c] = (val_targets[i, 0] == c).float()
    
    # 创建数据集
    train_dataset = TensorDataset(train_images, train_targets_onehot)
    val_dataset = TensorDataset(val_images, val_targets_onehot)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, checkpoint_dir=None):
    # 如果没有提供checkpoint_dir，默认使用03CNN算法\1空洞CNN目录
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join('03CNN算法', '1空洞CNN')
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    
    # 开始训练
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for images, targets in tqdm(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # 确保targets的维度正确
            if targets.dim() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item() * images.size(0)
        
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = images.to(device)
                targets = targets.to(device)
                
                # 确保targets的维度正确
                if targets.dim() == 4 and targets.size(1) == 1:
                    targets = targets.squeeze(1)
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 累计损失
                val_loss += loss.item() * images.size(0)
        
        # 计算平均验证损失
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f'Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f'Best model saved with val loss: {best_val_loss:.4f}')
    
    return model

# 定义交互式预测函数
def interactive_prediction():
    """交互式语义分割预测功能"""
    print("\n=== 语义分割预测工具 ===")
    print("本工具使用空洞CNN模型进行图像语义分割")
    print("注意：语义分割是图像处理任务，不是文本处理（不同于jieba库）")
    print("语义分割会将图像中的每个像素分类到不同类别")
    
    # 初始化模型
    num_classes = 21
    model = DilatedCNN(num_classes=num_classes).to(device)
    
    # 尝试加载已保存的模型
    model_path = os.path.join('03CNN算法', '1空洞CNN', 'best_model.pth')
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用未训练的模型进行预测")
    else:
        print("未找到已训练的模型，将使用未训练的模型进行预测")
        print("注意：未训练的模型分割结果可能不理想")
    
    # 设置模型为评估模式
    model.eval()
    
    # 使用cs.png进行测试
    image_size = 128
    try:
        # 读取cs.png文件
        image_path = os.path.join('03CNN算法', '1空洞CNN', 'cs.png')
        if os.path.exists(image_path):
            print(f"正在读取图像文件: {image_path}")
            # 打开图像并进行预处理
            img = Image.open(image_path).convert('RGB')
            img = img.resize((image_size, image_size))
            
            # 转换为numpy数组并进行标准化
            img_np = np.array(img).astype(np.float32) / 255.0
            # 调整维度顺序为(C, H, W)
            img_np = img_np.transpose(2, 0, 1)
            # 添加批次维度
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
            
            # 进行预测
            with torch.no_grad():
                output = model(img_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # 转换图像以便可视化（变回H, W, C顺序）
            demo_image_np = img_np.transpose(1, 2, 0)
            
            # 创建可视化图表
            plt.figure(figsize=(12, 6))
            
            # 显示原始图像
            plt.subplot(1, 2, 1)
            plt.imshow(demo_image_np)
            plt.title('原始图像 (cs.png)')
            plt.axis('off')
            
            # 显示分割结果
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask, cmap='jet')
            plt.title('语义分割结果')
            plt.axis('off')
            
            plt.tight_layout()
            
            # 保存结果
            result_path = os.path.join('03CNN算法', '1空洞CNN', 'segmentation_result.png')
            # 确保目录存在
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            plt.savefig(result_path)
            print(f"语义分割结果已保存到: {result_path}")
            
            # 尝试显示图像
            try:
                plt.show()
                print("图像显示完成")
            except:
                print("无法显示图像，请查看保存的图片文件")
        else:
            print(f"未找到图像文件: {image_path}")
            print("将生成随机测试图像进行演示")
            # 生成随机测试图像
            demo_image = torch.randn(1, 3, image_size, image_size).to(device)
            
            # 进行预测
            with torch.no_grad():
                output = model(demo_image)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # 转换图像以便可视化
            demo_image_np = demo_image.squeeze().cpu().numpy().transpose(1, 2, 0)
            # 归一化图像
            demo_image_np = (demo_image_np - demo_image_np.min()) / (demo_image_np.max() - demo_image_np.min() + 1e-8)
            
            # 创建可视化图表
            plt.figure(figsize=(12, 6))
            
            # 显示原始图像
            plt.subplot(1, 2, 1)
            plt.imshow(demo_image_np)
            plt.title('随机生成图像')
            plt.axis('off')
            
            # 显示分割结果
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask, cmap='jet')
            plt.title('语义分割结果')
            plt.axis('off')
            
            plt.tight_layout()
            
            # 保存结果
            result_path = os.path.join('03CNN算法', '1空洞CNN', 'segmentation_result.png')
            # 确保目录存在
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            plt.savefig(result_path)
            print(f"语义分割结果已保存到: {result_path}")
            
            # 尝试显示图像
            try:
                plt.show()
                print("图像显示完成")
            except:
                print("无法显示图像，请查看保存的图片文件")
        
        # 简单的用户交互说明
        print("\n语义分割预测完成！")
        print("\n关于语义分割的说明：")
        print("1. 语义分割是一种计算机视觉任务，不是文本处理任务")
        print("2. 与jieba库不同（jieba用于中文分词），语义分割处理的是图像数据")
        print("3. 语义分割的目标是识别图像中每个像素所属的类别")
        print("4. 常见应用包括自动驾驶、医学图像分析、卫星图像处理等")
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        print("请确保您的环境正确安装了所有必要的库")
    
    print("\n=== 预测工具结束 ===")

# 定义测试和可视化函数
def visualize_results(model, val_loader, num_samples=4):
    try:
        model.eval()
        
        # 选择几个样本进行可视化
        images, targets = next(iter(val_loader))
        images = images[:num_samples].to(device)
        
        with torch.no_grad():
            outputs = model(images)
            # 获取预测类别
            preds = torch.argmax(outputs, dim=1)
        
        # 将tensor转换为numpy数组并移至CPU
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        # 由于是随机数据，我们只取第一个通道作为标签可视化
        targets = targets[:num_samples, 0].cpu().numpy().transpose(0, 2, 3, 1)
        preds = preds.cpu().numpy()
        
        # 对随机图像进行归一化以便更好地可视化
        for i in range(num_samples):
            img = images[i]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            images[i] = img
        
        # 创建可视化图表
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i in range(num_samples):
            # 原始图像
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # 真实标签
            axes[i, 1].imshow(targets[i, :, :, 0], cmap='jet')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # 预测结果
            axes[i, 2].imshow(preds[i], cmap='jet')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('segmentation_results.png')
        print("可视化结果已保存到 segmentation_results.png")
        
        # 尝试显示图像，如果环境不支持则跳过
        try:
            plt.show()
        except:
            print("无法显示图像，已保存到文件")
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        print("跳过可视化步骤")

# 主函数
if __name__ == '__main__':
    # 超参数设置
    batch_size = 4
    image_size = 128  # 减小图像大小以提高运行速度
    num_classes = 21  # PASCAL VOC数据集有21个类别
    learning_rate = 1e-4
    num_epochs = 50  # 为了快速测试，只训练50个epoch
    
    # 创建模型
    model = DilatedCNN(num_classes=num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    try:
        print("正在创建模拟数据集...")
        # 加载数据
        train_loader, val_loader = get_dataloaders(batch_size=batch_size, image_size=image_size)
        
        print("开始训练模型...")
        # 训练模型
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )
        
        print("生成可视化结果...")
        # 可视化结果
        visualize_results(trained_model, val_loader)
        
        print("模型测试完成！")
        print("\n注意：")
        print("1. 此代码使用模拟数据进行测试，实际应用时需要替换为真实数据集")
        print("2. 为了快速测试，仅训练了1个epoch，实际应用时应增加训练轮数")
        print("3. 如需安装完整依赖，请使用: pip install -r requirements.txt")
        
        # 询问用户是否要进行预测
        try:
            choice = input("是否要进行语义分割预测？(y/n): ").strip().lower()
            if choice == 'y':
                interactive_prediction()
        except:
            # 如果无法使用input（例如在某些环境中），则直接启动交互式预测
            print("启动语义分割预测工具...")
            interactive_prediction()
        
    except Exception as e:
        print(f'运行过程中出错: {e}')
        print('\n模型基础测试:')
        # 为了防止程序因其他问题而完全失败，这里添加一个简单的模型测试
        try:
            x = torch.randn(1, 3, image_size, image_size).to(device)
            output = model(x)
            print(f'模型结构测试通过，输出形状: {output.shape}')
            print("模型可以正常运行，但可能需要安装完整依赖以获得更好的功能")
        except Exception as inner_e:
            print(f'模型测试也失败了: {inner_e}')
            print("请确保已正确安装PyTorch")
            
        # 无论训练是否成功，都尝试启动交互式预测功能
        try:
            print("\n尝试启动语义分割预测工具...")
            print("注意：由于训练过程中出现问题，分割结果可能不理想")
            interactive_prediction()
        except Exception as pred_e:
            print(f"启动预测工具时出错: {pred_e}")
            print("您可以直接调用interactive_prediction()函数来使用预测功能")
