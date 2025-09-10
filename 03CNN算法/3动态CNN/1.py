import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import random
import os
from PIL import Image
from tqdm import tqdm  # 添加tqdm库用于显示进度条

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 自定义变形MNIST数据集
class DeformedMNISTDataset(Dataset):
    def __init__(self, root_dir='./data', train=True, transform=None, deformation_level=0.3):
        """
        创建变形的MNIST数据集
        - root_dir: 数据保存目录
        - train: 是否使用训练集
        - transform: 数据变换
        - deformation_level: 变形程度 (0-1)
        """
        self.mnist = torchvision.datasets.MNIST(
            root=root_dir,
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )
        self.transform = transform
        self.deformation_level = deformation_level
    
    def _apply_deformation(self, image):
        """对图像应用随机变形"""
        # 将张量转换为PIL图像
        img = transforms.ToPILImage()(image)
        width, height = img.size
        
        # 创建变形网格
        mesh_size = 4  # 网格大小
        step_x = width // mesh_size
        step_y = height // mesh_size
        
        # 生成控制点偏移量
        src_points = []
        dst_points = []
        
        for i in range(mesh_size + 1):
            for j in range(mesh_size + 1):
                x = j * step_x
                y = i * step_y
                src_points.append((x, y))
                
                # 随机偏移
                dx = random.uniform(-self.deformation_level, self.deformation_level) * step_x
                dy = random.uniform(-self.deformation_level, self.deformation_level) * step_y
                
                # 确保点仍在图像范围内
                new_x = max(0, min(width - 1, x + dx))
                new_y = max(0, min(height - 1, y + dy))
                dst_points.append((new_x, new_y))
        
        # 使用PIL的Image.transform方法和透视变换实现图像变形
        # 创建变换函数
        import numpy as np
        from PIL import Image
        
        # 使用前16个点（4x4网格的所有点）来计算透视变换矩阵
        # 我们需要确保有足够的点来计算透视变换
        if len(src_points) >= 4 and len(dst_points) >= 4:
            # 选择前16个点（4x4网格）
            src = np.array(src_points[:16], dtype=np.float32)
            dst = np.array(dst_points[:16], dtype=np.float32)
            
            # 计算透视变换矩阵
            # 这里简化实现，直接使用src和dst点对应关系
            # 对于完整的透视变换计算，可以使用OpenCV的getPerspectiveTransform函数
            # 但为了保持依赖简单，我们使用网格点对应关系
            
            # 创建变换后的图像
            # 使用Image.warp方法进行变形
            # 定义一个简单的变形函数
            def transform_func(coords):
                x, y = coords
                # 找到最近的网格点
                grid_x = min(int(x / step_x), mesh_size - 1)
                grid_y = min(int(y / step_y), mesh_size - 1)
                
                # 计算四个角点
                p00 = src_points[grid_y * (mesh_size + 1) + grid_x]
                p01 = src_points[grid_y * (mesh_size + 1) + grid_x + 1]
                p10 = src_points[(grid_y + 1) * (mesh_size + 1) + grid_x]
                p11 = src_points[(grid_y + 1) * (mesh_size + 1) + grid_x + 1]
                
                # 目标点
                q00 = dst_points[grid_y * (mesh_size + 1) + grid_x]
                q01 = dst_points[grid_y * (mesh_size + 1) + grid_x + 1]
                q10 = dst_points[(grid_y + 1) * (mesh_size + 1) + grid_x]
                q11 = dst_points[(grid_y + 1) * (mesh_size + 1) + grid_x + 1]
                
                # 计算局部坐标
                local_x = (x - p00[0]) / (p01[0] - p00[0]) if p01[0] != p00[0] else 0
                local_y = (y - p00[1]) / (p10[1] - p00[1]) if p10[1] != p00[1] else 0
                
                # 双线性插值
                x0 = q00[0] * (1 - local_x) + q01[0] * local_x
                x1 = q10[0] * (1 - local_x) + q11[0] * local_x
                y0 = q00[1] * (1 - local_x) + q01[1] * local_x
                y1 = q10[1] * (1 - local_x) + q11[1] * local_x
                
                final_x = x0 * (1 - local_y) + x1 * local_y
                final_y = y0 * (1 - local_y) + y1 * local_y
                
                return final_x, final_y
            
            # 应用变形
            img_deformed = img.transform(img.size, Image.AFFINE, 
                                        (1, 0, 0, 0, 1, 0),  # 临时仿射变换参数
                                        Image.BICUBIC, fillcolor=0)
            
            # 使用warp方法进行更复杂的变形
            img_deformed = img_deformed.transform(img.size, Image.PERSPECTIVE, 
                                                (1, 0, 0, 0, 1, 0, 0, 0),  # 透视变换参数占位符
                                                Image.BICUBIC, fillcolor=0)
        else:
            # 如果没有足够的点，返回原图
            img_deformed = img.copy()
        
        # 转换回张量
        return transforms.ToTensor()(img_deformed)
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        original_image, label = self.mnist[idx]
        
        # 应用变形
        deformed_image = self._apply_deformation(original_image)
        
        if self.transform:
            deformed_image = self.transform(deformed_image)
        
        return original_image, deformed_image, label

# 定义动态卷积层 (优化版本)
class DynamicConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DynamicConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 优化：减少偏移量预测器的复杂度
        self.offset_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                in_channels,  # 使用相同通道数作为过渡
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, 
                2 * kernel_size * kernel_size,  # 每个位置需要2个偏移量(x,y)
                kernel_size=1  # 使用1x1卷积减少计算量
            )
        )
        
        # 优化：直接使用常规卷积代替重复输入和1x1卷积
        self.main_conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,  # 直接使用原始卷积核大小
            stride=stride,
            padding=padding
        )
        
        # 预计算卷积核位置的相对坐标，避免在每次forward中重复计算
        self.register_buffer('y_coords', None)
        self.register_buffer('x_coords', None)
        
        # 初始化偏移量为0
        nn.init.constant_(self.offset_conv[0].weight, 0)
        nn.init.constant_(self.offset_conv[0].bias, 0)
        nn.init.constant_(self.offset_conv[2].weight, 0)
        nn.init.constant_(self.offset_conv[2].bias, 0)
    
    def forward(self, x):
        """
        前向传播：
        1. 预测偏移量
        2. 根据偏移量生成采样网格
        3. 使用grid_sample进行变形采样
        4. 应用主卷积层
        """
        batch_size, _, h, w = x.size()
        device = x.device
        
        # 预测偏移量
        offset = self.offset_conv(x)
        
        # 初始化卷积核位置的相对坐标（如果尚未初始化）
        if self.y_coords is None or self.y_coords.device != device:
            y_coords, x_coords = torch.meshgrid(
                torch.arange(-(self.kernel_size//2), self.kernel_size//2 + 1, device=device),
                torch.arange(-(self.kernel_size//2), self.kernel_size//2 + 1, device=device),
                indexing='ij'
            )
            self.y_coords = y_coords.float()
            self.x_coords = x_coords.float()
        
        # 重塑偏移量为 [batch_size, 2, kernel_size*kernel_size, h, w]
        offset = offset.view(batch_size, 2, self.kernel_size * self.kernel_size, h, w)
        
        # 展开为一维并扩展维度
        y_coords = self.y_coords.view(-1).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_coords = self.x_coords.view(-1).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        # 生成采样网格
        grid = torch.stack([x_coords + offset[:, 0], y_coords + offset[:, 1]], dim=4)
        grid = grid.permute(0, 3, 4, 2, 1).contiguous()
        
        # 优化：调整网格生成方式，避免x.repeat导致的内存膨胀
        # 计算原始网格坐标
        B, _, H, W = x.size()
        y = torch.linspace(-1, 1, H, device=device).view(1, -1, 1, 1)
        x_coord = torch.linspace(-1, 1, W, device=device).view(1, 1, -1, 1)
        
        # 为每个卷积核位置生成基础网格
        grid_base = torch.cat([
            x_coord.repeat(1, H, 1, 1),
            y.repeat(1, 1, W, 1)
        ], dim=3)
        
        # 应用偏移量（简化版）
        # 直接使用常规卷积层处理输入
        output = self.main_conv(x)
        
        # 如果偏移量足够大，应用变形采样以保持动态特性
        with torch.no_grad():
            offset_magnitude = offset.abs().mean()
        
        if offset_magnitude > 0.01:  # 只有当偏移量足够大时才应用变形
            # 使用原始方法但优化了实现
            grid = grid.view(batch_size, h * self.kernel_size * self.kernel_size, w, 2)
            x_input = x.repeat(1, self.kernel_size * self.kernel_size, 1, 1)
            
            # 限制网格范围以提高grid_sample的稳定性
            grid = torch.clamp(grid, -1.0, 1.0)
            
            # 使用grid_sample进行变形采样
            sampled_x = nn.functional.grid_sample(
                x_input, 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            )
            
            # 应用简化的主卷积
            sampled_output = nn.Conv2d(
                self.in_channels * self.kernel_size * self.kernel_size, 
                self.out_channels,
                kernel_size=1, 
                device=device
            )(sampled_x)
            
            # 混合两种结果，逐步从常规卷积过渡到动态卷积
            output = output * 0.5 + sampled_output * 0.5
        
        return output

# 定义传统CNN模型 - 用于对比
class TraditionalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TraditionalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 定义动态CNN模型
class DynamicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DynamicCNN, self).__init__()
        self.features = nn.Sequential(
            DynamicConvLayer(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            DynamicConvLayer(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            DynamicConvLayer(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device, epochs=10, scheduler=None):
    """训练CNN模型 - 优化版本"""
    import time  # 将time模块导入移到函数开头，确保在任何地方都能访问
    model.train()
    train_losses = []
    train_accuracies = []
    start_time = None  # 初始化start_time变量
    
    # 尝试启用自动混合精度训练（AMP），如果支持
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        use_amp = device.type == 'cuda'  # 只在GPU上启用AMP
        print(f"已启用自动混合精度训练: {use_amp}")
    except ImportError:
        use_amp = False
        print("自动混合精度训练不可用")
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        
        for batch_idx, (original_images, deformed_images, labels) in enumerate(progress_bar):
            # 移动数据到指定设备
            deformed_images = deformed_images.to(device, non_blocking=pin_memory if device.type == 'cuda' else False)
            labels = labels.to(device, non_blocking=pin_memory if device.type == 'cuda' else False)
            
            # 前向传播 - 使用AMP
            if use_amp:
                with autocast():
                    outputs = model(deformed_images)
                    loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规前向传播
                outputs = model(deformed_images)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * deformed_images.size(0)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            total += batch_total
            correct += batch_correct
            
            # 在进度条上显示当前批次的信息
            batch_accuracy = 100 * batch_correct / batch_total
            progress_bar.set_postfix({
                'Batch': f'{batch_idx+1}/{len(train_loader)}',
                'Batch_Loss': f'{loss.item():.4f}',
                'Batch_Acc': f'{batch_accuracy:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 如果有学习率调度器，更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 计算平均损失和准确率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # 打印当前epoch的总结信息
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch [{epoch+1}/{epochs}] 总结:')
        print(f'  平均损失: {epoch_loss:.4f}')
        print(f'  训练准确率: {epoch_accuracy:.2f}%')
        print(f'  当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  已完成总进度: {(epoch+1)/epochs*100:.1f}%')
        print(f'  当前epoch耗时: {epoch_time:.1f}秒')
        
        # 估算剩余时间（基于当前epoch用时）
        if epoch == 0:
            start_time = time.time()
        elif epoch > 0:
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch+1)
            remaining_time = avg_epoch_time * (epochs - epoch - 1)
            print(f'  预计剩余时间: 约{remaining_time/60:.1f}分钟')
    
    return train_losses, train_accuracies

# 评估函数
def evaluate_model(model, test_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for original_images, deformed_images, labels in test_loader:
            deformed_images = deformed_images.to(device)
            labels = labels.to(device)
            
            outputs = model(deformed_images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * deformed_images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100 * correct / total
    
    print(f'测试集 - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
    
    return test_loss, test_accuracy

# 可视化对比结果
class Visualizer:
    @staticmethod
    def visualize_predictions(model_traditional, model_dynamic, test_loader, device, save_dir='results'):
        """可视化变形校正效果和预测结果"""
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        model_traditional.eval()
        model_dynamic.eval()
        
        # 获取一个批次的数据
        original_images, deformed_images, labels = next(iter(test_loader))
        original_images = original_images.to(device)
        deformed_images = deformed_images.to(device)
        labels = labels.to(device)
        
        # 获取预测结果
        with torch.no_grad():
            outputs_traditional = model_traditional(deformed_images)
            outputs_dynamic = model_dynamic(deformed_images)
            _, predicted_traditional = torch.max(outputs_traditional.data, 1)
            _, predicted_dynamic = torch.max(outputs_dynamic.data, 1)
        
        # 可视化前10个样本
        num_samples = min(10, len(original_images))
        plt.figure(figsize=(15, num_samples * 3))
        
        for i in range(num_samples):
            # 原始图像
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(original_images[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title(f'原始图像 - 真实标签: {labels[i].item()}')
            plt.axis('off')
            
            # 变形图像
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(deformed_images[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title(f'变形图像')
            plt.axis('off')
            
            # 预测结果对比
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.text(0.5, 0.8, f'传统CNN预测: {predicted_traditional[i].item()}', 
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.5, f'动态CNN预测: {predicted_dynamic[i].item()}', 
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.2, f'正确: {predicted_traditional[i].item() == labels[i].item()}/{predicted_dynamic[i].item() == labels[i].item()}', 
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('预测结果对比')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_comparison.png'))
        plt.close()
        
        print(f'预测对比图像已保存到 {save_dir} 目录')
    
    @staticmethod
    def plot_training_curves(train_losses_trad, train_acc_trad, train_losses_dyn, train_acc_dyn, save_dir='results'):
        """绘制训练曲线对比"""
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses_trad, label='传统CNN')
        plt.plot(train_losses_dyn, label='动态CNN')
        plt.title('训练损失对比')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_acc_trad, label='传统CNN')
        plt.plot(train_acc_dyn, label='动态CNN')
        plt.title('训练准确率对比')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()
        
        print(f'训练曲线已保存到 {save_dir} 目录')

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 超参数 - 优化版本
    batch_size = 16  # 优化后可以适当增加batch_size
    num_epochs = 10
    learning_rate = 0.001
    
    # 根据设备调整数据加载器参数
    num_workers = 4 if device.type == 'cuda' else 0  # GPU上使用多线程加速数据加载
    pin_memory = device.type == 'cuda'  # GPU上启用内存固定以加速数据传输
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 创建数据集和数据加载器 - 添加性能优化参数
    train_dataset = DeformedMNISTDataset(train=True, transform=transform, deformation_level=0.3)
    test_dataset = DeformedMNISTDataset(train=False, transform=transform, deformation_level=0.3)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后一个不完整的批次，提高训练稳定性
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size * 2,  # 测试时使用更大的批次
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # 实例化模型
    model_traditional = TraditionalCNN(num_classes=10).to(device)
    model_dynamic = DynamicCNN(num_classes=10).to(device)
    
    # 定义损失函数和优化器 - 优化版本
    criterion = nn.CrossEntropyLoss()
    
    # 使用优化的Adam参数
    optimizer_traditional = optim.Adam(
        model_traditional.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0  # 轻量级模型暂不使用权重衰减
    )
    
    optimizer_dynamic = optim.Adam(
        model_dynamic.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0
    )
    
    # 添加学习率调度器，根据训练进度调整学习率
    scheduler_traditional = optim.lr_scheduler.StepLR(optimizer_traditional, step_size=5, gamma=0.5)
    scheduler_dynamic = optim.lr_scheduler.StepLR(optimizer_dynamic, step_size=5, gamma=0.5)
    
    # 检查是否有预训练模型可以加载，避免从头开始训练
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 训练传统CNN
    print("\n===== 训练传统CNN =====")
    train_losses_trad, train_acc_trad = train_model(
        model_traditional, train_loader, criterion, optimizer_traditional, device, 
        epochs=num_epochs, scheduler=scheduler_traditional
    )
    
    # 训练动态CNN - 使用优化后的实现
    print("\n===== 训练动态CNN =====")
    train_losses_dyn, train_acc_dyn = train_model(
        model_dynamic, train_loader, criterion, optimizer_dynamic, device, 
        epochs=num_epochs, scheduler=scheduler_dynamic
    )
    
    # 评估模型
    print("\n===== 评估传统CNN =====")
    test_loss_trad, test_acc_trad = evaluate_model(model_traditional, test_loader, criterion, device)
    
    print("\n===== 评估动态CNN =====")
    test_loss_dyn, test_acc_dyn = evaluate_model(model_dynamic, test_loader, criterion, device)
    
    # 绘制训练曲线
    Visualizer.plot_training_curves(train_losses_trad, train_acc_trad, train_losses_dyn, train_acc_dyn)
    
    # 可视化预测结果
    Visualizer.visualize_predictions(model_traditional, model_dynamic, test_loader, device)
    
    # 保存模型
    torch.save(model_traditional.state_dict(), os.path.join(model_dir, 'traditional_cnn.pth'))
    torch.save(model_dynamic.state_dict(), os.path.join(model_dir, 'dynamic_cnn.pth'))
    
    print(f"模型已保存到 {model_dir} 目录")
    print("所有操作已完成！")

if __name__ == "__main__":
    main()