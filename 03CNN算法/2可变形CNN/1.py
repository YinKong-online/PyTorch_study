import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import librosa
import librosa.display
import soundfile as sf
import random

# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "NSimSun"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 实现可变形卷积模块
class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformConv2d, self).__init__()
        
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.kernel_size = kernel_size
        
        # 普通卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        
        # 偏移量卷积层，输出2*kernel_size*kernel_size个通道（每个位置的x和y偏移）
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        
        # 初始化偏移量为0
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        # 保存卷积核大小，避免重复计算
        self.num_points = kernel_size * kernel_size
    
    def forward(self, x):
        # 计算偏移量
        offset = self.offset_conv(x)
        batch_size, _, h, w = offset.shape
        in_channels = x.shape[1]
        
        # 重塑偏移量 - 2×k×k×h×w
        offset = offset.view(batch_size, 2, self.kernel_size, self.kernel_size, h, w)
        
        # 生成采样点坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device) * self.stride[0],
            torch.arange(w, device=device) * self.stride[1],
            indexing='ij'
        )
        
        # 生成卷积核相对坐标网格
        ky, kx = torch.meshgrid(
            torch.arange(self.kernel_size, device=device) - (self.kernel_size - 1) / 2,
            torch.arange(self.kernel_size, device=device) - (self.kernel_size - 1) / 2,
            indexing='ij'
        )
        
        # 扩展维度
        y_coords = y_coords.view(1, 1, h, w).expand(batch_size, self.kernel_size, self.kernel_size, -1, -1)
        x_coords = x_coords.view(1, 1, h, w).expand(batch_size, self.kernel_size, self.kernel_size, -1, -1)
        
        ky = ky.view(1, self.kernel_size, self.kernel_size, 1, 1).expand(batch_size, -1, -1, h, w)
        kx = kx.view(1, self.kernel_size, self.kernel_size, 1, 1).expand(batch_size, -1, -1, h, w)
        
        # 计算采样点的绝对坐标
        sample_y = y_coords + ky + offset[:, 0, ...]
        sample_x = x_coords + kx + offset[:, 1, ...]
        
        # 将坐标归一化到[-1, 1]范围
        h_original = x.shape[2]
        w_original = x.shape[3]
        sample_y = 2 * sample_y / (h_original - 1) - 1
        sample_x = 2 * sample_x / (w_original - 1) - 1
        
        # 组合坐标并调整维度以匹配grid_sample要求
        grid = torch.stack((sample_x, sample_y), dim=-1)
        grid = grid.view(batch_size * self.num_points, h, w, 2)
        
        # 复制输入以匹配网格批次维度
        x_expanded = x.repeat_interleave(self.num_points, dim=0)
        
        # 使用grid_sample进行采样
        sampled_x = nn.functional.grid_sample(x_expanded, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 重塑采样结果
        sampled_x = sampled_x.view(batch_size, self.num_points, in_channels, h, w)
        
        # 合并卷积核维度，保持输入通道数不变
        sampled_x = sampled_x.sum(dim=1)
        
        # 应用普通卷积
        output = self.conv(sampled_x)
        
        return output

# 可变形卷积块
class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformConvBlock, self).__init__()
        
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.deform_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 音频生成器网络
class AudioGenerator(nn.Module):
    def __init__(self, input_dim=100, n_mels=80, seq_len=128):
        super(AudioGenerator, self).__init__()
        self.input_dim = input_dim
        self.n_mels = n_mels
        self.seq_len = seq_len
        
        # 投影层将潜在向量映射到特征图
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 128 * seq_len // 8 * n_mels // 8),
            nn.LeakyReLU(0.2)
        )
        
        # 转置卷积层和可变形卷积层交替使用
        self.deconv_layers = nn.Sequential(
            # 上采样到 seq_len//4 x n_mels//4
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 可变形卷积块
            DeformConvBlock(64, 64),
            
            # 上采样到 seq_len//2 x n_mels//2
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 可变形卷积块
            DeformConvBlock(32, 32),
            
            # 上采样到 seq_len x n_mels
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # 可变形卷积块
            DeformConvBlock(16, 16),
            
            # 最终输出层
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 将输出限制在[-1, 1]范围
        )
    
    def forward(self, z):
        # z: [batch_size, input_dim]
        x = self.projection(z)
        # 重塑为特征图: [batch_size, 128, seq_len//8, n_mels//8]
        x = x.view(-1, 128, self.seq_len // 8, self.n_mels // 8)
        # 经过转置卷积和可变形卷积
        x = self.deconv_layers(x)
        # 输出形状: [batch_size, 1, seq_len, n_mels]
        return x

# 音频判别器网络
class AudioDiscriminator(nn.Module):
    def __init__(self, n_mels=80, seq_len=128):
        super(AudioDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 输入层
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 可变形卷积块
            DeformConvBlock(16, 32),
            
            # 下采样
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 可变形卷积块
            DeformConvBlock(64, 128),
            
            # 下采样
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 可变形卷积块
            DeformConvBlock(256, 256),
        )
        
        # 计算特征图最终大小
        self.final_size = (seq_len // 8) * (n_mels // 8) * 256
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.final_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch_size, 1, seq_len, n_mels]
        x = self.model(x)
        # 展平特征图
        x = x.view(-1, self.final_size)
        # 输出判别结果
        x = self.output_layer(x)
        return x

# 音频数据集类
class AudioDataset(Dataset):
    def __init__(self, data_dir, n_mels=80, seq_len=128, sample_rate=22050, n_fft=1024, hop_length=256):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # 获取所有音频文件
        self.audio_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".wav") or file.endswith(".mp3"):
                    self.audio_files.append(os.path.join(root, file))
        
        # 如果没有音频文件，使用合成数据
        if len(self.audio_files) == 0:
            print("警告: 未找到音频文件，将使用合成数据进行训练")
            self.use_synthetic = True
            self.synthetic_data_size = 1000  # 合成数据大小
        else:
            self.use_synthetic = False
            print(f"找到 {len(self.audio_files)} 个音频文件")
    
    def __len__(self):
        if self.use_synthetic:
            return self.synthetic_data_size
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        if self.use_synthetic:
            # 生成合成的梅尔谱图
            mel = np.random.randn(self.seq_len, self.n_mels).astype(np.float32)
            # 添加一些结构使其看起来更像真实的梅尔谱图
            for i in range(5):
                start = np.random.randint(0, self.seq_len - 20)
                end = start + 20
                freq_start = np.random.randint(0, self.n_mels - 10)
                freq_end = freq_start + 10
                mel[start:end, freq_start:freq_end] += np.random.randn(20, 10) * 0.5
        else:
            # 加载真实音频文件
            file_path = self.audio_files[idx]
            try:
                # 加载音频文件
                y, sr = librosa.load(file_path, sr=self.sample_rate)
                
                # 确保音频长度足够
                if len(y) < self.hop_length * self.seq_len:
                    # 如果音频太短，填充零
                    y = np.pad(y, (0, max(0, self.hop_length * self.seq_len - len(y))))
                else:
                    # 如果音频太长，随机选择一段
                    start = np.random.randint(0, len(y) - self.hop_length * self.seq_len + 1)
                    y = y[start:start + self.hop_length * self.seq_len]
                
                # 计算梅尔谱图
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft,
                                                  hop_length=self.hop_length, n_mels=self.n_mels)
                
                # 转换为对数刻度
                mel = librosa.power_to_db(S, ref=np.max)
                
                # 归一化到[-1, 1]
                mel = (mel - mel.min()) / (mel.max() - mel.min()) * 2 - 1
            except Exception as e:
                print(f"加载音频文件 {file_path} 失败: {e}")
                # 如果加载失败，生成随机数据
                mel = np.random.randn(self.seq_len, self.n_mels).astype(np.float32)
        
        # 添加通道维度并转换为张量
        mel = torch.from_numpy(mel).unsqueeze(0)
        return mel

# 训练函数
def train_model(generator, discriminator, dataloader, num_epochs=50, lr=0.0002, beta1=0.5, beta2=0.999):
    # 移动模型到设备
    generator.to(device)
    discriminator.to(device)
    
    # 定义优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # 定义损失函数
    criterion = nn.BCELoss()
    
    # 固定的潜在向量用于生成样本
    fixed_noise = torch.randn(8, generator.input_dim, device=device)
    
    # 创建结果保存目录
    os.makedirs("03CNN算法/2可变形CNN/generated_samples", exist_ok=True)
    os.makedirs("03CNN算法/2可变形CNN/models", exist_ok=True)
    
    # 最佳损失值
    best_g_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        d_losses = []
        g_losses = []
        
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, real_mels in enumerate(dataloader):
                # 准备数据
                real_mels = real_mels.to(device)
                batch_size = real_mels.size(0)
                
                # 标签
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                
                # ---------------------训练判别器---------------------
                optimizer_d.zero_grad()
                
                # 判别真实样本
                real_output = discriminator(real_mels)
                d_loss_real = criterion(real_output, real_labels)
                
                # 生成假样本
                noise = torch.randn(batch_size, generator.input_dim, device=device)
                fake_mels = generator(noise)
                
                # 判别假样本
                fake_output = discriminator(fake_mels.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                # 总判别器损失
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()
                
                # ---------------------训练生成器---------------------
                optimizer_g.zero_grad()
                
                # 生成假样本
                fake_mels = generator(noise)
                
                # 通过判别器
                fake_output = discriminator(fake_mels)
                
                # 生成器损失
                g_loss = criterion(fake_output, real_labels)
                g_loss.backward()
                optimizer_g.step()
                
                # 记录损失
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                
                # 更新进度条
                pbar.set_postfix({"D Loss": np.mean(d_losses), "G Loss": np.mean(g_losses)})
                pbar.update(1)
        
        # 保存最佳生成器模型
        current_g_loss = np.mean(g_losses)
        if current_g_loss < best_g_loss:
            best_g_loss = current_g_loss
            torch.save(generator.state_dict(), "03CNN算法/2可变形CNN/models/best_generator.pth")
            print(f"Epoch {epoch+1}: 保存最佳生成器模型，损失值: {best_g_loss:.4f}")
        
        # 每5个epoch生成并保存样本
        if (epoch + 1) % 5 == 0:
            generator.eval()
            with torch.no_grad():
                fake_mels = generator(fixed_noise)
                save_generated_samples(fake_mels, epoch + 1)
            generator.train()
        
        # 打印 epoch 信息
        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}")
    
    # 保存最终模型
    torch.save(generator.state_dict(), "03CNN算法/2可变形CNN/models/final_generator.pth")
    torch.save(discriminator.state_dict(), "03CNN算法/2可变形CNN/models/final_discriminator.pth")
    print("训练完成，模型已保存")

# 保存生成的样本
def save_generated_samples(fake_mels, epoch):
    fake_mels = fake_mels.cpu().numpy()
    
    # 创建图像
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, mel in enumerate(fake_mels[:8]):
        # 去除通道维度
        mel = mel.squeeze(0)
        
        # 在轴上显示梅尔谱图
        librosa.display.specshow(mel, x_axis='time', y_axis='mel', ax=axes[i])
        axes[i].set_title(f"生成样本 #{i+1}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f"03CNN算法/2可变形CNN/generated_samples/epoch_{epoch}.png")
    plt.close()

# 生成音频函数
def generate_audio_from_model(generator, num_samples=5, sample_rate=22050, n_fft=1024, hop_length=256):
    # 创建输出目录
    os.makedirs("03CNN算法/2可变形CNN/generated_audio", exist_ok=True)
    
    # 加载最佳模型
    try:
        generator.load_state_dict(torch.load("03CNN算法/2可变形CNN/models/best_generator.pth", map_location=device))
        print("加载最佳生成器模型成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    generator.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            # 生成随机潜在向量
            noise = torch.randn(1, generator.input_dim, device=device)
            
            # 生成梅尔谱图
            fake_mel = generator(noise)
            fake_mel = fake_mel.squeeze().cpu().numpy()
            
            # 将梅尔谱图转换回音频
            # 反归一化
            fake_mel = (fake_mel + 1) * 0.5  # 转换到[0, 1]范围
            
            # 应用逆变换
            # 注意：从梅尔谱图到音频是一个不适定问题，这里使用 Griffin-Lim 算法近似
            # 由于我们没有真实的梅尔滤波器组参数，这里使用 librosa 的默认参数
            # 创建梅尔滤波器组
            mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=generator.n_mels)
            
            # 将梅尔谱图转换回线性谱图
            S = librosa.feature.inverse.mel_to_stft(fake_mel, sr=sample_rate, n_fft=n_fft)
            
            # 使用 Griffin-Lim 算法恢复音频
            y = librosa.griffinlim(S, hop_length=hop_length)
            
            # 保存音频文件
            audio_path = f"03CNN算法/2可变形CNN/generated_audio/generated_{i+1}.wav"
            sf.write(audio_path, y, sample_rate)
            
            # 保存梅尔谱图图像
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(fake_mel, x_axis='time', y_axis='mel', sr=sample_rate)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"生成的梅尔谱图 #{i+1}")
            plt.tight_layout()
            plt.savefig(f"03CNN算法/2可变形CNN/generated_audio/spectrogram_{i+1}.png")
            plt.close()
            
            print(f"已生成音频文件: {audio_path}")

# 主函数
def main():
    # 超参数设置
    n_mels = 80
    seq_len = 128
    batch_size = 16
    num_epochs = 20  # 可根据需要调整
    lr = 0.0002
    input_dim = 100
    
    # 数据集路径（这里使用当前目录作为示例）
    data_dir = "."
    
    # 创建数据集和数据加载器
    dataset = AudioDataset(data_dir, n_mels=n_mels, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 创建模型
    generator = AudioGenerator(input_dim=input_dim, n_mels=n_mels, seq_len=seq_len)
    discriminator = AudioDiscriminator(n_mels=n_mels, seq_len=seq_len)
    
    # 打印模型结构摘要
    print("生成器模型结构:")
    print(generator)
    # 计算模型参数量
    num_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"生成器参数量: {num_params:,}\n")
    
    print("判别器模型结构:")
    print(discriminator)
    # 计算模型参数量
    num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"判别器参数量: {num_params:,}")
    
    # 训练模型
    print("\n开始训练模型...")
    train_model(generator, discriminator, dataloader, num_epochs=num_epochs, lr=lr)
    
    # 生成音频
    print("\n生成音频样本...")
    generate_audio_from_model(generator)
    
    print("\n所有操作已完成！")

# 如果作为主程序运行
if __name__ == "__main__":
    main()