import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from typing import List, Tuple, Dict
import math
import seaborn as sns
import sys
sys.path.append(r'E:\20250711电机小论文')

class AdaptiveWaveletLayer(nn.Module):
    """自适应小波层"""
    def __init__(self, signal_length, wavelet_scales, config):
        super(AdaptiveWaveletLayer, self).__init__()
        self.signal_length = signal_length
        self.wavelet_scales = wavelet_scales
        self.config = config
        
        # 可学习的小波基函数参数
        self.wavelet_weights = nn.Parameter(torch.randn(wavelet_scales, signal_length) * 0.1)
        
        # 尺度参数
        self.scale_params = nn.Parameter(torch.ones(wavelet_scales))
        
        # 小波选择权重
        self.wavelet_selection = nn.Parameter(torch.ones(len(config.WAVELET_TYPES)) / len(config.WAVELET_TYPES))
        
        # 频率权重
        self.freq_weights = nn.Parameter(torch.ones(signal_length // 2))
        
        print(f"AdaptiveWaveletLayer初始化: signal_length={signal_length}, scales={wavelet_scales}")

    def forward(self, x):
        """
        前向传播 - 处理五维输入
        
        输入 x 形状: [batch_size, 5, signal_length]
        """
        batch_size, num_channels, signal_length = x.size()
        
        # 多尺度小波变换
        wavelet_coeffs = []
        
        for scale in range(self.wavelet_scales):
            # 自适应小波基函数
            wavelet_basis = self.wavelet_weights[scale] * self.scale_params[scale]
            wavelet_basis = wavelet_basis.unsqueeze(0).unsqueeze(0)  # [1, 1, signal_length]
            
            # 对每个通道进行小波变换
            channel_coeffs = []
            for channel_idx in range(num_channels):
                channel_input = x[:, channel_idx:channel_idx+1, :]
                coeffs = F.conv1d(channel_input, wavelet_basis, padding='same')
                channel_coeffs.append(coeffs.squeeze(1))
            
            # 堆叠每个通道的系数
            scale_coeffs = torch.stack(channel_coeffs, dim=1)
            wavelet_coeffs.append(scale_coeffs)
        
        # 堆叠所有尺度的系数
        wavelet_features = torch.stack(wavelet_coeffs, dim=2)  
        # 形状变为 [batch_size, num_channels, scales, signal_length]
        
        return wavelet_features



    def get_wavelet_info(self):
        """获取小波信息"""
        return {
            'scales': self.wavelet_scales,
            'scale_params': self.scale_params.detach().cpu().numpy(),
            'wavelet_selection': F.softmax(self.wavelet_selection, dim=0).detach().cpu().numpy()
        }

class ContrastiveLearningModule(nn.Module):
    """对比学习模块"""
    def __init__(self, feature_dim, config):
        super(ContrastiveLearningModule, self).__init__()
        self.feature_dim = feature_dim
        self.config = config
        
        # 特征投影层
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, config.CAAF_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.CAAF_HIDDEN_DIM, config.CAAF_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.CAAF_HIDDEN_DIM // 2, 128)  # 投影到128维
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        print(f"ContrastiveLearningModule初始化: feature_dim={feature_dim}")

    def forward(self, features, labels=None):
        """前向传播"""
        # 特征投影
        projected_features = self.projection_head(features)
        
        # L2归一化
        projected_features = F.normalize(projected_features, dim=1)
        
        if labels is not None and self.training:
            # 计算对比损失
            contrast_loss = self.compute_contrastive_loss(projected_features, labels)
            return projected_features, contrast_loss
        
        return projected_features, torch.tensor(0.0)


    def compute_contrastive_loss(self, features, labels):
        """计算对比损失"""
        batch_size = features.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # 移除对角线
        mask = mask - torch.eye(batch_size, device=features.device)
        
        # 计算正样本和负样本的logits
        positive_logits = similarity_matrix * mask
        negative_logits = similarity_matrix * (1 - mask)
        
        # 计算损失
        positive_exp = torch.exp(positive_logits)
        negative_exp = torch.exp(negative_logits)
        
        # 避免除零
        positive_sum = torch.sum(positive_exp, dim=1, keepdim=True) + 1e-8
        negative_sum = torch.sum(negative_exp, dim=1, keepdim=True) + 1e-8
        
        loss = -torch.log(positive_sum / (positive_sum + negative_sum))
        
        return torch.mean(loss)

class MultiscaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, scales, feature_dim, config, num_channels=5):
        """
        初始化多尺度特征融合模块
        
        参数:
        - scales: 小波尺度数
        - feature_dim: 特征维度
        - config: 配置对象
        - num_channels: 输入通道数，默认为5
        """
        super(MultiscaleFeatureFusion, self).__init__()
        self.scales = scales
        self.feature_dim = feature_dim
        self.num_channels = num_channels
        self.config = config
        
        # 不同尺度和通道的特征处理
        self.scale_processors = nn.ModuleList()
        for i in range(scales):
            channel_processors = nn.ModuleList()
            for j in range(num_channels):
                channel_processors.append(nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(128)
                ))
            self.scale_processors.append(channel_processors)
        
        # 注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(scales * num_channels * 64 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, scales * num_channels),
            nn.Softmax(dim=1)
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(scales * num_channels * 64 * 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)
        )

    def forward(self, wavelet_features):
        """
        前向传播 - 处理多尺度小波特征
        输入 wavelet_features 形状: [batch_size, num_channels, scales, signal_length]
        """
        batch_size = wavelet_features.size(0)
        
        # 处理每个尺度和通道的特征
        scale_features = []
        for scale_idx in range(self.scales):
            channel_features = []
            for channel_idx in range(self.num_channels):
                # 提取当前尺度和通道的输入
                scale_channel_input = wavelet_features[:, channel_idx, scale_idx, :]
                scale_channel_input = scale_channel_input.unsqueeze(1)  # 添加通道维度 [batch_size, 1, signal_length]
                
                # 处理特征并展平
                scale_channel_feat = self.scale_processors[scale_idx][channel_idx](scale_channel_input)  # [batch_size, 64, 128]
                scale_channel_feat = scale_channel_feat.flatten(1)  # 展平为 [batch_size, 64*128=8192]
                channel_features.append(scale_channel_feat)
            
            # 拼接当前尺度的所有通道特征 [batch_size, 5*8192=40960]
            scale_feat = torch.cat(channel_features, dim=1)
            scale_features.append(scale_feat)
        
        # 拼接所有尺度的特征 [batch_size, 5*40960=204800]
        all_features = torch.cat(scale_features, dim=1)
        
        # 计算注意力权重 [batch_size, 25]（25=5尺度×5通道）
        attention_weights = self.attention_weights(all_features)
        
        # ########## 修正：按子向量分割并加权 ##########
        per_channel_dim = 64 * 128  # 每个（尺度-通道）对的特征维度
        weighted_features = []
        
        for i in range(self.scales * self.num_channels):
            # 计算当前子向量的起始和结束索引
            start = i * per_channel_dim
            end = start + per_channel_dim
            
            # 提取子向量并乘以对应的注意力权重（广播权重到子向量维度）
            channel_feat = all_features[:, start:end]  # [batch_size, 8192]
            attn_weight = attention_weights[:, i:i+1]  # [batch_size, 1]
            weighted_feat = channel_feat * attn_weight  # [batch_size, 8192]
            
            weighted_features.append(weighted_feat)
        
        # 拼接所有加权后的子向量 [batch_size, 25*8192=204800]
        fused_features = torch.cat(weighted_features, dim=1)
        
        # 融合特征 [batch_size, 512]
        output = self.fusion_layer(fused_features)
        
        return output, attention_weights



class MLWCN(nn.Module):
    """多尺度提升小波对比网络"""
    def __init__(self, config):
        super(MLWCN, self).__init__()
        self.config = config
        
        # 配置参数
        self.signal_length = config.SIGNAL_LENGTH
        self.wavelet_scales = config.MLWCN_SCALES
        self.output_dim = 256
        
        # 自适应小波层
        self.adaptive_wavelet = AdaptiveWaveletLayer(
            self.signal_length,  
            self.wavelet_scales,  
            config
        )
        
        # 多尺度特征融合
        self.multiscale_fusion = MultiscaleFeatureFusion(
            self.wavelet_scales,
            self.output_dim,
            config
        )
        
        # 对比学习模块
        self.contrastive_learning = ContrastiveLearningModule(
            self.output_dim,
            config
        )
        
        # 正交约束模块
        self.orthogonal_constraint = nn.Parameter(torch.eye(self.wavelet_scales))
        
        print(f"MLWCN初始化完成，输出维度: {self.output_dim}")

    def forward(self, x, labels=None):
        """前向传播"""
        # 确保输入是三维的 [batch_size, 5, signal_length]
        if x.dim() != 3 or x.size(1) != 5:
            raise ValueError(f"输入张量维度错误，期望 [batch_size, 5, signal_length]，实际为 {x.size()}")
        
        # 直接使用self.adaptive_wavelet进行小波变换
        wavelet_features = self.adaptive_wavelet(x)
        
        # 多尺度特征融合
        fused_features, attention_weights = self.multiscale_fusion(wavelet_features)
        
        # 对比学习
        contrast_features, contrast_loss = self.contrastive_learning(fused_features, labels)
        
        # 计算其他损失
        orthogonal_loss = self.compute_orthogonal_loss()
        adaptive_loss = self.compute_adaptive_loss()
        multiscale_loss = self.compute_multiscale_loss(attention_weights)
        
        # 构建损失字典
        losses = {
            'contrast_loss': contrast_loss,
            'orthogonal_loss': orthogonal_loss,
            'adaptive_loss': adaptive_loss,
            'multiscale_loss': multiscale_loss
        }
        
        return fused_features, losses


    def compute_orthogonal_loss(self):
        """计算正交约束损失"""
        # 小波基函数的正交性约束
        wavelet_weights = self.adaptive_wavelet.wavelet_weights
        
        # 计算Gram矩阵
        gram_matrix = torch.matmul(wavelet_weights, wavelet_weights.T)
        
        # 正交约束：Gram矩阵应该接近单位矩阵
        identity = torch.eye(self.wavelet_scales, device=wavelet_weights.device)
        orthogonal_loss = F.mse_loss(gram_matrix, identity)
        
        return orthogonal_loss

    def compute_adaptive_loss(self):
        """计算自适应损失"""
        # 尺度参数的稀疏性约束
        scale_params = self.adaptive_wavelet.scale_params
        adaptive_loss = torch.sum(torch.abs(scale_params))
        
        return adaptive_loss * 0.01  # 缩放因子

    def compute_multiscale_loss(self, attention_weights):
        """计算多尺度损失"""
        # 注意力权重的平衡性约束
        uniform_weights = torch.ones_like(attention_weights) / self.wavelet_scales
        multiscale_loss = F.kl_div(
            F.log_softmax(attention_weights, dim=1),
            uniform_weights,
            reduction='batchmean'
        )
        
        return multiscale_loss

    def get_wavelet_analysis(self):
        """获取小波分析信息"""
        wavelet_info = self.adaptive_wavelet.get_wavelet_info()
        
        analysis = {
            'wavelet_info': wavelet_info,
            'orthogonal_matrix': self.orthogonal_constraint.detach().cpu().numpy(),
            'model_params': {
                'scales': self.wavelet_scales,
                'signal_length': self.signal_length,
                'output_dim': self.output_dim
            }
        }
        
        return analysis
#
# 测试
if __name__ == "__main__":
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from configs.config import config
    from data.university_of_ottawa_loader import UniversityOfOttawaDataLoader
    from preprocess.data_preprocessor import DataPreprocessor
    import seaborn as sns
    import sys
    sys.path.append(r'E:\20250711电机小论文')
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    # 数据集根目录
    DATA_ROOT = r"E:\20250711电机小论文\data"

    # 初始化数据加载器和预处理器
    data_loader = UniversityOfOttawaDataLoader(DATA_ROOT, config)
    preprocessor = DataPreprocessor(config)


    # 加载数据集
    data_list, labels, metadata = data_loader.load_dataset(
        use_csv=True,  # 使用CSV文件
        conditions=[0, 1]  # 加载空载和负载工况
    )

    # 数据预处理
    processed_vibration_1, processed_microphone, processed_vibration_2, \
    processed_vibration_3, processed_temperature, processed_labels, processed_metadata = preprocessor.process_dataset(
        data_list, labels, metadata, apply_augmentation=False
    )

    # 初始化模型
    model = MLWCN(config)
    print("\n模型架构:")
    print(model)

    # 选择部分数据进行测试
    def prepare_test_data(accelerometer_1, microphone, 
                       accelerometer_2, accelerometer_3, 
                       temperature, labels, num_samples=32):
        """准备测试数据"""
        # 使用numpy数组确保正确索引
        accelerometer_1 = np.array(accelerometer_1)
        microphone = np.array(microphone)
        accelerometer_2 = np.array(accelerometer_2)
        accelerometer_3 = np.array(accelerometer_3)
        temperature = np.array(temperature)
        labels = np.array(labels)

        # 随机选择样本索引
        indices = np.random.choice(len(accelerometer_1), num_samples, replace=False)
        
        # 使用索引选择数据
        test_signals = np.stack([
            accelerometer_1[indices],
            microphone[indices],
            accelerometer_2[indices],
            accelerometer_3[indices],
            temperature[indices]
        ], axis=1)
        
        test_labels = labels[indices]
        
        # 转换为张量
        test_signals = torch.tensor(test_signals, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
        
        return test_signals, test_labels


        
    # 准备测试数据（以例）
    test_signals, test_labels = prepare_test_data(  
    processed_vibration_1,  # accelerometer_1  
    processed_microphone,   # microphone  
    processed_vibration_2,  # accelerometer_2  
    processed_vibration_3,  # accelerometer_3  
    processed_temperature,  # temperature  
    processed_labels  
)  
    print(f"\n测试数据形状: {test_signals.shape}")
    print(f"测试标签形状: {test_labels.shape}")

    # 前向传播
    with torch.no_grad():
        fused_features, losses = model(test_signals, test_labels)
        
        print("\n损失详情:")
        for loss_name, loss_value in losses.items():
            print(f"{loss_name}: {loss_value.item():.4f}")
        
        print(f"\n融合特征形状: {fused_features.shape}")

    # 可视化部分

    # 1. 小波分析可视化
    wavelet_analysis = model.get_wavelet_analysis()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("小波尺度参数")
    plt.plot(wavelet_analysis['wavelet_info']['scale_params'], marker='o')
    plt.xlabel("尺度索引")
    plt.ylabel("尺度参数值")

    plt.subplot(1, 3, 2)
    plt.title("小波类型选择权重")
    plt.bar(config.WAVELET_TYPES, wavelet_analysis['wavelet_info']['wavelet_selection'])
    plt.xlabel("小波类型")
    plt.ylabel("选择权重")
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    plt.title("正交约束矩阵")
    sns.heatmap(wavelet_analysis['orthogonal_matrix'], cmap='viridis')
    plt.tight_layout()
    plt.savefig('wavelet_analysis.png')
    plt.close()

    # 2. 特征分布可视化
    plt.figure(figsize=(10, 6))
    plt.title("融合特征分布")
    for label in torch.unique(test_labels):
        mask = test_labels == label
        features_subset = fused_features[mask]
        label_name = data_loader.fault_names[label.item()]
        plt.scatter(
            features_subset[:, 0].numpy(), 
            features_subset[:, 1].numpy(), 
            label=label_name, 
            alpha=0.7
        )
    plt.xlabel("特征维度1")
    plt.ylabel("特征维度2")
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_distribution.png')
    plt.close()

    # 3. 注意力权重可视化 - 修正
    plt.figure(figsize=(10, 6))
    plt.title("多尺度特征融合注意力权重")

    # 获取最后一层的权重参数（即softmax之前的线性层输出）
    # 注意：这里需要访问权重参数而不是模块本身
    attention_weights = model.multiscale_fusion.attention_weights[2].weight.detach().numpy()

    # 转置以匹配原始代码的形状假设
    sns.heatmap(
        attention_weights.T, 
        cmap='viridis', 
        cbar_kws={'label': '注意力权重'}
    )
    plt.xlabel("特征维度")
    plt.ylabel("尺度×通道")
    plt.tight_layout()
    plt.savefig('attention_weights.png')
    plt.close()

    print("\n可视化结果已保存: wavelet_analysis.png, feature_distribution.png, attention_weights.png")
