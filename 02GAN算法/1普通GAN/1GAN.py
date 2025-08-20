# GAN算法实现
# 项目：生成一串特定数据做密码(11位，6数字，5字母，其中1个大写)
# 目标：生成一串符合规则的密码
# 步骤：先给D一些正确的密码(以字典格式,密码:真假)，
# 让G生成一些密码，然后让D来判断这些数据是否正确
# 练习一段时间：
# 调用G生成密码，若生成密码无法转ASCII码,print('密码生成失败'),
# 若生成密码可以转ASCII码, 打印让我看
# 调用D，我手动输入一些密码，有真有假，看D的判断结果(1为真，0为假)
# 结构：
# D 判别器
# G 生成器
# 损失函数：BCELoss(交叉熵损失函数)
# 优化器
# 练习

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import string
from sklearn.utils import shuffle

# 数据集(错误密码)
fake_data = [
    '825g19Print', 'f25264kkJnt', '825H525r4kt', '82lk195r1d2',
    '8252S4kkJnt', '825H52Lrikt', '82lk191r1d2', '8252645kJnt',
    '8)5H525rikt', '82lkg9Pr1d2', '8252FFkkJnt', '82sH525rikt',
    '8252d4kkJnt', '825Hsd5rikt', '8ilk19Pr1d2', '82df64kkJnt',
    '825264k5Jnt', '825H525r5kt', '82lk191r1d2', '8252645kJnt',
    'quajkjsisjj', '012d4567891', '01g34567891', '0123jkij891',
    '01234sd7891', '0123s567891', '012sss67891', '0123f567891',
    'ajksl154544', '15151223151', 'uiuiui15151', '12345678901'
]
fake_labels = [0] * len(fake_data)  # 假密码标签为0

# 程序生成符合规则的样本
def generate_valid_password():
    digits = [random.choice(string.digits) for _ in range(6)]
    letters = []
    upper_added = False
    for _ in range(5):
        if not upper_added and random.random() < 0.2:
            letters.append(random.choice(string.ascii_uppercase))
            upper_added = True
        else:
            letters.append(random.choice(string.ascii_lowercase))
    combined = digits + letters
    random.shuffle(combined)
    return ''.join(combined)

# 生成更多数据
valid_data = [generate_valid_password() for _ in range(100)]  # 生成100个符合规则的密码
valid_labels = [1] * len(valid_data)  # 真密码标签为1

# 合并数据集
data = valid_data + fake_data
labels = valid_labels + fake_labels
data, labels = shuffle(data, labels, random_state=42)

# 数据预处理函数
def string_to_tensor(s):
    ascii_values = [ord(c) for c in s]
    # 归一化到[0,1]，基于可打印ASCII范围32-126
    normalized = [(c - 32) / (126 - 32) for c in ascii_values]
    return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

def tensor_to_string(tensor):
    tensor = tensor.squeeze()  # 移除batch维度
    scaled = tensor * (126 - 32) + 32  # 反归一化到32-126
    ascii_codes = torch.clamp(scaled, 32, 126).cpu().numpy().astype(int)
    return ''.join([chr(c) for c in ascii_codes])

# 有效性检查：11位，6个数字，5个字母，其中1个大写字母
def is_valid_password(password):
    if len(password) != 11:
        return False
    num_digits = sum(c.isdigit() for c in password)
    num_letters = sum(c.isalpha() for c in password)
    has_upper = any(c.isupper() for c in password)
    return num_digits == 6 and num_letters == 5 and has_upper

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 生成器模型
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 11),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = Discriminator().to(device)
G = Generator().to(device)
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 自定义模型保存路径
model_dir = 'GAN算法'  # 你可以根据需要修改这个目录路径
os.makedirs(model_dir, exist_ok=True)  # 如果目录不存在则创建

# 预训练模型路径
discriminator_model_path = os.path.join(model_dir, '1GAN-D.pth')
generator_model_path = os.path.join(model_dir, '1GAN-G.pth')

# 检查是否有预训练模型
if os.path.exists(discriminator_model_path) and os.path.exists(generator_model_path):
    D.load_state_dict(torch.load(discriminator_model_path))
    G.load_state_dict(torch.load(generator_model_path))
    print(f"加载了预训练模型：{discriminator_model_path}, {generator_model_path}")
else:
    print("没有找到预训练模型，从头开始训练")

# 准备真实数据
real_samples = []
for s in data:
    tensor = string_to_tensor(s).squeeze(0)  # 移除batch维度
    real_samples.append(tensor)
real_samples = torch.stack(real_samples).to(device)

# 训练循环
num_epochs = 10000
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, len(real_samples), batch_size):
        # 真实数据批次
        real_batch = real_samples[i:i+batch_size]
        real_labels = torch.ones(real_batch.size(0), 1, device=device)
        
        # 生成假数据
        noise = torch.randn(real_batch.size(0), 100, device=device)
        fake_batch = G(noise)
        fake_labels = torch.zeros(real_batch.size(0), 1, device=device)
        
        # 训练判别器
        D.train()
        d_optimizer.zero_grad()
        
        real_outputs = D(real_batch)
        d_loss_real = criterion(real_outputs, real_labels)
        
        fake_outputs = D(fake_batch.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # 训练生成器
        G.train()
        g_optimizer.zero_grad()
        
        gen_outputs = D(fake_batch)
        g_loss = criterion(gen_outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    
    # 每100次保存模型
    if (epoch + 1) % 1000 == 0:
        torch.save(D.state_dict(), discriminator_model_path)
        torch.save(G.state_dict(), generator_model_path)
        print(f"模型已保存：Epoch {epoch} 到 {discriminator_model_path} 和 {generator_model_path}")

# 测试生成器
def generate_password():
    G.eval()
    with torch.no_grad():
        noise = torch.randn(1, 100).to(device)
        generated = G(noise)
        password = tensor_to_string(generated)
        if is_valid_password(password):
            return password
        else:
            return None  # 如果无效，返回None

# 测试判别器
def test_discriminator():
    D.eval()
    password = input("请输入测试密码: ")
    if len(password) != 11:
        print("无效长度")
        return
    tensor = string_to_tensor(password).to(device)
    with torch.no_grad():
        output = D(tensor)
    print(f"判别结果: {output.item():.4f} (接近1为真，接近0为假)")

# 示例调用
print("\n生成测试:")
for i in range(10):
    pwd = generate_password()
    print(pwd)
    print('生成的密码:', pwd if pwd else '密码生成失败')

print("\n判别器测试:")
test_discriminator()