# 条件GAN 算法
# 项目：违禁词检测
# 目标：句子是否具有违禁词
# 步骤：先给D一些有违禁词的句子,(！重点：不告诉违禁点)
# 让G生成一些句子，然后让D来判断这些数据是否正确
# 结构：
# D 判别器
# G 生成器
# 损失函数
# 优化器
# 练习

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

# 读取JSON数据
def load_json_data(file_path):
    try:
        # 使用绝对路径确保文件能被找到
        abs_path = os.path.join(os.path.dirname(__file__), file_path)
        if not os.path.exists(abs_path):
            print(f"文件 {abs_path} 不存在")
            return {}
        with open(abs_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的JSON格式")
        return {}
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return {}

# 加载真实和虚假数据
real_data = load_json_data('2(1).json')  # 真实数据（有违禁词）
fake_data = load_json_data('2(0).json')  # 虚假数据（无违禁词）

# 打印数据加载情况
print(f"加载了 {len(real_data)} 条真实数据")
print(f"加载了 {len(fake_data)} 条虚假数据")

# 简单的文本预处理和转换
def text_to_tensor(text, vocab_size, max_length=100):
    tensor = torch.zeros(max_length, dtype=torch.long)
    for i, char in enumerate(text[:max_length]):
        # 将字符映射到词汇表范围内的整数
        tensor[i] = ord(char) % vocab_size
    return tensor

# 准备训练数据
def prepare_training_data(data, vocab_size, max_length=100, topic_label=0):
    if not data:
        return torch.tensor([]), torch.tensor([])

    texts = list(data.keys())
    topics = [topic_label] * len(texts)

    sentences = []
    for text in texts:
        tensor = text_to_tensor(text, vocab_size, max_length)
        sentences.append(tensor)

    return torch.stack(sentences), torch.tensor(topics, dtype=torch.long)

# 准备训练数据
real_sentences, real_topics = prepare_training_data(real_data, vocab_size, max_length=100, topic_label=0)
fake_sentences_json, fake_topics_json = prepare_training_data(fake_data, vocab_size, max_length=100, topic_label=1)

# 合并真实和虚假数据用于训练
all_sentences = torch.cat([real_sentences, fake_sentences_json]) if len(real_sentences) > 0 and len(fake_sentences_json) > 0 else real_sentences
all_topics = torch.cat([real_topics, fake_topics_json]) if len(real_topics) > 0 and len(fake_topics_json) > 0 else real_topics

print(f"总训练数据量: {len(all_sentences)}")

# Condition Encoder: 用于将话题标签转换为嵌入向量
class ConditionEncoder(nn.Module):
    def __init__(self, num_topics, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_topics, embed_dim)
        
    def forward(self, topic_labels):
        return self.embedding(topic_labels)  # 例：8个话题，128维嵌入 num_topics=8, embed_dim=128

# Generator: 生成器
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, vocab_size):
        super().__init__()
        self.condition_encoder = ConditionEncoder(num_topics=8, embed_dim=128)
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LSTM(512, 512),  # 加入LSTM生成序列
            nn.Linear(512, vocab_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, z, topic_labels):
        # 拼接噪声和条件
        cond = self.condition_encoder(topic_labels) 
        input = torch.cat([z, cond], dim=1) 
        return self.model(input)  # 输出词表概率分布

# Discriminator: 判别器
class Discriminator(nn.Module):
    def __init__(self, vocab_size, condition_dim):
        super().__init__()
        self.condition_encoder = ConditionEncoder(num_topics=8, embed_dim=128)
        
        self.feature_extractor = nn.Sequential(
            nn.Embedding(vocab_size, 256),
            nn.LSTM(256, 256, bidirectional=True),
            nn.Linear(512, 128)
        )
        
        self.joint_judge = nn.Sequential(
            nn.Linear(128 + condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),  # 真实性判断
            nn.Sigmoid()
        )
        
        self.topic_classifier = nn.Linear(128, 8)  # 辅助话题分类

    def forward(self, sentences, topic_labels):
        # 特征提取
        features = self.feature_extractor(sentences)
        
        # 条件融合
        cond = self.condition_encoder(topic_labels)
        combined = torch.cat([features, cond], dim=1)
        
        # 双任务输出
        validity = self.joint_judge(combined)
        topic_prob = F.softmax(self.topic_classifier(features), dim=1)
        
        return validity, topic_prob

# 计算梯度惩罚（WGAN-GP）
def gradient_penalty(real_sentences, fake_sentences, discriminator):
    # 计算真实和假数据的插值
    alpha = torch.randn(real_sentences.size(0), 1, 1).to(real_sentences.device)
    interpolated = alpha * real_sentences + (1 - alpha) * fake_sentences
    interpolated = interpolated.requires_grad_(True)
    
    # 计算判别器的输出
    validity, _ = discriminator(interpolated)
    
    # 计算梯度
    gradients = torch.autograd.grad(outputs=validity, inputs=interpolated, grad_outputs=torch.ones_like(validity), create_graph=True, retain_graph=True)[0]
    
    # 计算梯度惩罚
    grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gp_loss = ((grad_norm - 1) ** 2).mean()
    return gp_loss

# 计算损失函数
def compute_loss(real_validity, fake_validity, real_topic, pred_topic, real_sentences, fake_sentences, discriminator):
    # 对抗损失
    adv_loss = torch.mean(fake_validity) - torch.mean(real_validity)
    
    # 话题分类损失（强制条件匹配）
    cls_loss = F.cross_entropy(pred_topic, real_topic)
    
    # 梯度惩罚（WGAN-GP）
    gp_loss = gradient_penalty(real_sentences, fake_sentences, discriminator)
    
    return adv_loss + 0.5*cls_loss + 10*gp_loss

# 生成阶段添加词替换
def apply_replace(text, replace_dict):
    # 替换违禁词
    for word, replacements in replace_dict.items():
        if word in text:
            text = text.replace(word, replacements[0])  # 默认使用第一个同义词
    return text

def generate_with_camouflage(generator, z, topics):
    raw_output = generator(z, topics)
    
    # 对违禁词进行同义词替换
    replace_dict = {
        "枪支": ["运动器材", "金属管件"],
        "毒品": ["白色粉末", "不明物质"]
    }
    return apply_replace(raw_output, replace_dict)

# 违禁词检测示例（简单实现）
def rule_based_detect(sentence):
    banned_words = ["枪支", "毒品"]  # 简单例子，实际情况可以从数据库或词典获取
    detected = [word for word in banned_words if word in sentence]
    return detected

# 计算句子与目标话题的相似度（使用cosine similarity）
def cosine_similarity(sentence_embedding, topic_embedding):
    # 使用sklearn计算cosine相似度，假设已将句子嵌入到合适的向量空间
    return sk_cosine_similarity([sentence_embedding], [topic_embedding])[0][0]

# 训练过程
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

# 初始化生成器和判别器
noise_dim = 100  # 随机噪声的维度
condition_dim = 128  # 话题标签嵌入的维度
vocab_size = 5000  # 假设词汇表的大小为5000
generator = Generator(noise_dim=noise_dim, condition_dim=condition_dim, vocab_size=vocab_size)
discriminator = Discriminator(vocab_size=vocab_size, condition_dim=condition_dim)

# 损失函数和优化器
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

batch_size = 32
# 设置训练过程的主循环
for epoch in range(100):
    # 如果没有足够的训练数据，则跳过
    if len(all_sentences) < batch_size:
        print("训练数据不足，无法进行训练")
        break
    
    # 随机选择一个批次的训练数据
    idx = torch.randperm(len(all_sentences))[:batch_size]
    batch_sentences = all_sentences[idx]
    batch_topics = all_topics[idx]
    
    # 随机噪声
    z = torch.randn(batch_size, noise_dim)
    
    # 生成假数据
    fake_sentences = generator(z, batch_topics)

    # 训练判别器
    real_validity, real_topic_prob = discriminator(batch_sentences, batch_topics)
    fake_validity, fake_topic_prob = discriminator(fake_sentences, batch_topics)
    
    # 计算损失
    loss = compute_loss(real_validity, fake_validity, batch_topics, fake_topic_prob, batch_sentences, fake_sentences, discriminator)
    
    # 更新判别器
    optimizer_D.zero_grad()
    loss.backward()
    optimizer_D.step()

    # 训练生成器（生成器的目标是让判别器认为生成的句子是“真实”的）
    optimizer_G.zero_grad()
    fake_validity, _ = discriminator(fake_sentences, batch_topics)
    g_loss = -torch.mean(fake_validity)  # 生成器希望判别器认为假数据是真实的
    g_loss.backward()
    optimizer_G.step()

    # 每个epoch输出训练情况
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/100] - Generator Loss: {g_loss.item():.4f} - Discriminator Loss: {loss.item():.4f}")
