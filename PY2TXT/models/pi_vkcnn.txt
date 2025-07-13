import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
import seaborn as sns
import sys
sys.path.append(r'E:\20250711电机小论文')
import matplotlib.pyplot as plt

class VariationalKernel(nn.Module):
    """物理信息约束的变分核卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_size, config):
        super(VariationalKernel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = min(kernel_size, config.SIGNAL_LENGTH // 2)  
        self.config = config
        
        # 变分参数初始化
        self.weight_mu = Parameter(torch.randn(out_channels, in_channels, self.kernel_size) * 0.01)
        self.weight_log_sigma = Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.001)
        
        self.bias_mu = Parameter(torch.randn(out_channels) * 0.01)
        self.bias_log_sigma = Parameter(torch.randn(out_channels) * 0.01)
        
        # 物理约束参考核
        self.register_buffer('physics_kernel', self._init_physics_kernel())
        self.register_buffer('freq_weights', self._init_freq_weights())

    def _init_physics_kernel(self):
        """
        基于电机故障物理模型的参考核。。。动态频率权重初始化 
        （）
        特点:
        - 基于故障频谱特征  
        - 自适应频率敏感性  
        - 考虑信号的谱特性 
        """
        physics_kernel = torch.zeros(self.out_channels, self.in_channels, self.kernel_size)
        
        # 故障模式参数
        fault_modes = {
            'bearing_spall': {'freq': 5, 'modulation': 0.5},
            'gear_tooth_crack': {'freq': 25, 'modulation': 0.3},
            'rotor_eccentricity': {'freq': 10, 'modulation': 0.4},
            'stator_winding': {'freq': 15, 'modulation': 0.2},  
            'voltage_unbalance': {'freq': 20, 'modulation': 0.3}    
        }
        
        for i in range(self.out_channels):
            mode_selector = list(fault_modes.keys())[i % len(fault_modes)]
            mode_params = fault_modes[mode_selector]
            
            t = torch.linspace(-1, 1, self.kernel_size)
            
            # 物理约束的时频建模
            envelope = torch.exp(-t**2 / 0.2)
            base_carrier = torch.cos(2 * math.pi * mode_params['freq'] * t)
            
            # 故障特征调制
            modulated_carrier = base_carrier * (1 + mode_params['modulation'] * t)
            
            # 多通道模拟
            for j in range(self.in_channels):
                phase_shift = j * math.pi / self.in_channels
                physics_kernel[i, j, :] = envelope * torch.cos(
                    2 * math.pi * mode_params['freq'] * t + phase_shift
                ) * modulated_carrier
        
        # 强约束归一化
        physics_kernel = (physics_kernel - physics_kernel.mean()) / physics_kernel.std()
        
        return physics_kernel


    def _init_freq_weights(self):
        """初始化频率权重- 基于故障频谱特征  
        - 自适应频率敏感性  
        - 考虑信号的谱特性  """
        
        # 频率权重的自适应构建  
        freq_weights = torch.ones(self.kernel_size)  
        
        # 故障特征频率区间  
        low_freq_end = self.kernel_size // 4  
        mid_freq_end = self.kernel_size // 2  
        high_freq_start = self.kernel_size * 3 // 4  
        
        # 多区间权重策略  
        freq_weights[:low_freq_end] = 3.0     # 低频关键故障特征  
        freq_weights[low_freq_end:mid_freq_end] = 2.0  # 中频次要特征  
        freq_weights[mid_freq_end:high_freq_start] = 1.5  # 过渡频带  
        freq_weights[high_freq_start:] = 1.0   # 高频噪声抑制  
        
        # 高斯平滑，减少权重跳变  
        gaussian_window = self._generate_gaussian_window()  
        freq_weights = F.conv1d(  
            freq_weights.view(1, 1, -1),   
            gaussian_window.view(1, 1, -1),   
            padding='same'  
        ).squeeze()  
        
        return freq_weights  
    def _generate_gaussian_window(self, window_size=5, sigma=1.0):  
        """  
        生成高斯平滑窗口  
        
        参数:  
        - window_size: 窗口大小  
        - sigma: 高斯分布标准差  
        """  
        x = torch.linspace(-(window_size // 2), window_size // 2, window_size)  
        window = torch.exp(-(x ** 2) / (2 * sigma ** 2))  
        return window / window.sum()  

    def forward(self, x):  
        """  
        前向传播的增强版本  
        
        新增特性:  
        - 自适应重参数化  
        - 动态截断  
        - 稳定性增强  
        """  
        if x.dim() == 2:  
            x = x.unsqueeze(1)  # [batch_size, signal_length] -> [batch_size, 1, signal_length]  
        
        # 计算填充  
        padding = self.kernel_size // 2 

        # 自适应重参数化  
        weight_sigma = torch.clamp(torch.exp(self.weight_log_sigma), min=1e-6)  
        bias_sigma = torch.clamp(torch.exp(self.bias_log_sigma), min=1e-6)  
        
        # 改进的随机性采样  
        weight_epsilon = torch.randn_like(self.weight_mu)  
        bias_epsilon = torch.randn_like(self.bias_mu)  
        
        # 带截断的重参数化  
        weight = self.weight_mu + weight_sigma * weight_epsilon  
        bias = self.bias_mu + bias_sigma * bias_epsilon  
        
        # 正则化权重  
        weight = F.relu(weight)  # 非负约束  
        
        # 卷积操作（带掩码）  
        mask = torch.sigmoid(weight)  # 软掩码  
        output = F.conv1d(x, self.weight_mu, self.bias_mu, padding=padding)    
        
        # 损失计算  
        kl_loss = self._compute_kl_loss()  
        physics_loss = self._compute_physics_loss()  
        causality_loss = self._compute_causality_loss()  
        
        return output, kl_loss, physics_loss, causality_loss  

    def _compute_kl_loss(self):  
        """  
        KL散度损失的改进版  
        
        新特性:  
        - 自适应权重  
        - 稀疏性约束  
        """  
        # 基础KL散度  
        weight_kl = -0.5 * torch.sum(  
            1 + self.weight_log_sigma   
            - self.weight_mu.pow(2)   
            - self.weight_log_sigma.exp()  
        )  
        
        bias_kl = -0.5 * torch.sum(  
            1 + self.bias_log_sigma   
            - self.bias_mu.pow(2)   
            - self.bias_log_sigma.exp()  
        )  
        
        # 稀疏性正则  
        sparsity_reg = torch.mean(torch.abs(self.weight_mu))  
        
        return weight_kl + bias_kl + 0.1 * sparsity_reg  


    def _compute_physics_loss(self):  
        """  
        物理约束损失的增强版  
        
        新特性:  
        - 多尺度频率损失  
        - 谱相似性  
        """  
        # 时域物理损失  
        time_physics_loss = F.mse_loss(self.weight_mu, self.physics_kernel)  
        
        # 频域变换  
        weight_fft = torch.fft.fft(self.weight_mu, dim=-1)  
        physics_fft = torch.fft.fft(self.physics_kernel, dim=-1)  
        
        # 多尺度频率损失  
        freq_physics_loss = torch.mean(  
            self.freq_weights * torch.abs(weight_fft - physics_fft) ** 2  
        )  
        
        # 谱相关性  
        spectral_similarity = 1 - F.cosine_similarity(  
            weight_fft.real.flatten(),   
            physics_fft.real.flatten(),   
            dim=0  
        )  
        
        return time_physics_loss + freq_physics_loss + 0.5 * spectral_similarity

    def _compute_causality_loss(self):  
        """  
        因果性损失的优化版  
        
        新特性:  
        - 软约束  
        - 自适应阈值  
        - 多尺度因果分析  
        """  
        kernel_center = self.kernel_size // 2  
        causality_loss = 0.0  
        
        # 自适应阈值  
        adaptive_threshold = self.config.CAUSALITY_THRESHOLD * torch.std(self.weight_mu)  
        
        # 多尺度因果分析  
        for scale in [1, 2, 4]:  
            for i in range(kernel_center // scale):  
                future_weight = self.weight_mu[:, :, kernel_center + i*scale + 1:]  
                if future_weight.numel() > 0:  
                    # 软约束  
                    causality_loss += F.relu(  
                        torch.mean(torch.abs(future_weight)) - adaptive_threshold  
                    ).sum()  
        
        return causality_loss

class PI_VKCNN(nn.Module):
    """物理信息约束的变分核卷积网络"""
    def __init__(self, config):
        super(PI_VKCNN, self).__init__()
        self.config = config
        
        # 多尺度变分核卷积层
        self.vk_layers = nn.ModuleList()
        channels = config.PI_VKCNN_CHANNELS
        kernels = config.PI_VKCNN_KERNELS
        
        for i in range(len(kernels)):
            self.vk_layers.append(
                VariationalKernel(
                    channels[i], 
                    channels[i+1], 
                    kernels[i], 
                    config
                )
            )
        
        # 网络层配置
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm1d(channels[i+1]) for i in range(len(kernels))
        ])
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = channels[-1]

        # 可视化存储  
        self.layer_visualizations = [] 

    def forward(self, x):
        """前向传播"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        total_kl_loss = 0
        total_physics_loss = 0
        total_causality_loss = 0
        
        # 通过变分核卷积层
        for i, (vk_layer, bn_layer) in enumerate(zip(self.vk_layers, self.batch_norm)):
            x, kl_loss, physics_loss, causality_loss = vk_layer(x)
            
            x = bn_layer(x)
            x = self.activation(x)
            x = self.dropout(x)
           
            # 可视化存储  
            self.layer_visualizations.append(x.clone())

            if i < len(self.vk_layers) - 1:
                x = F.max_pool1d(x, kernel_size=2, stride=2)
            
            total_kl_loss += kl_loss
            total_physics_loss += physics_loss
            total_causality_loss += causality_loss
        
        # 自适应池化
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        
        # 构建损失字典
        losses = {
            'kl_loss': total_kl_loss,
            'physics_loss': total_physics_loss,
            'causality_loss': total_causality_loss
        }
        
        return x, losses

    def _store_layer_visualization(self, vk_layer, layer_idx):  
        """存储层可视化信息"""  
        layer_vis = {  
            'layer_idx': layer_idx,  
            'weight_mu': vk_layer.weight_mu.detach().cpu().numpy(),  
            'physics_kernel': vk_layer.physics_kernel.detach().cpu().numpy(),  
            'freq_weights': vk_layer.freq_weights.detach().cpu().numpy()  
        }  
        self.layer_visualizations.append(layer_vis)

    def visualize_physics_interpretation(self, save_path='pi_vkcnn_interpretation'):  
        """  
        可视化物理解释  
        
        生成多子图:  
        1. 变分核权重热力图  
        2. 物理参考核对比  
        3. 频率权重分布  
        4. 物理相似度雷达图  
        """  
        import matplotlib.pyplot as plt  
        import seaborn as sns  

        plt.figure(figsize=(20, 15))  
        
        # 总体物理相似度  
        physics_similarities = []  
        
        # 为每一层变分核创建单独的可视化  
        for i, vk_layer in enumerate(self.vk_layers):  
            plt.figure(figsize=(20, 15))  
            
            # 变分核权重热力图 - 对3D张量进行处理  
            plt.subplot(2, 2, 1)  
            # 选择第一个通道的权重进行可视化  
            weight_mu_2d = vk_layer.weight_mu[0].detach().cpu().numpy()  
            sns.heatmap(weight_mu_2d, cmap='viridis',   
                        xticklabels=False,   
                        yticklabels=False,   
                        cbar_kws={'label': 'Weight Value'})  
            plt.title(f'Layer {i} - Variational Kernel Weights')  
            
            # 物理参考核对比  
            plt.subplot(2, 2, 2)  
            physics_kernel_2d = vk_layer.physics_kernel[0].detach().cpu().numpy()  
            plt.plot(weight_mu_2d, label='Variational Kernel')  
            plt.plot(physics_kernel_2d, label='Physics Kernel')  
            plt.title(f'Layer {i} - Physics Kernel Comparison')  
            plt.legend()  
            
            # 频率权重分布  
            plt.subplot(2, 2, 3)  
            freq_weights = vk_layer.freq_weights.detach().cpu().numpy()  
            plt.plot(freq_weights)  
            plt.title(f'Layer {i} - Frequency Weights')  
            plt.xlabel('Frequency Bin')  
            plt.ylabel('Weight')  
            
            # 计算物理相似度  
            similarity = np.corrcoef(  
                vk_layer.weight_mu.flatten().detach().cpu().numpy(),  
                vk_layer.physics_kernel.flatten().detach().cpu().numpy()  
            )[0, 1]  
            physics_similarities.append(similarity)  
            
            # 保存每层的可视化图像  
            plt.tight_layout()  
            plt.savefig(f'{save_path}_layer_{i}_physics_interpretation.png')  
            plt.close()  
    
        # 绘制物理相似度雷达图  
        plt.figure(figsize=(10, 10))  
        angles = np.linspace(0, 2*np.pi, len(physics_similarities), endpoint=False)  
        physics_similarities = np.concatenate((physics_similarities, [physics_similarities[0]]))  
        angles = np.concatenate((angles, [angles[0]]))  
        
        plt.polar(angles, physics_similarities)  
        plt.fill(angles, physics_similarities, alpha=0.25)  
        plt.title('Physics Similarity across Layers')  
        plt.savefig(f'{save_path}_physics_similarities_radar.png')  
        plt.close()  

      

    def _plot_kernel_heatmap(self, layer_vis, layer_idx):  
        """变分核权重热力图"""  
        weight_mu_2d = layer_vis['weight_mu'][0].squeeze()  
        plt.title(f'Layer {layer_idx} - Variational Kernel Weights')  
        sns.heatmap(weight_mu_2d, cmap='viridis')  

    def _plot_physics_kernel_comparison(self, layer_vis, layer_idx):  
        """物理参考核对比"""  
        plt.title(f'Layer {layer_idx} - Physics Kernel Comparison')  
        plt.plot(layer_vis['weight_mu'].squeeze(), label='Variational Kernel')  
        plt.plot(layer_vis['physics_kernel'].squeeze(), label='Physics Kernel')  
        plt.legend()  

    def _plot_frequency_weights(self, layer_vis, layer_idx):  
        """频率权重分布"""  
        plt.title(f'Layer {layer_idx} - Frequency Weights')  
        plt.plot(layer_vis['freq_weights'])  
        plt.xlabel('Frequency Bin')  
        plt.ylabel('Weight')  

    def _plot_physics_similarity_radar(self, similarities):  
        """物理相似度雷达图"""  
        plt.title('Physics Similarity across Layers')  
        angles = np.linspace(0, 2*np.pi, len(similarities), endpoint=False)  
        similarities = np.concatenate((similarities, [similarities[0]]))  
        angles = np.concatenate((angles, [angles[0]]))  
        
        plt.polar(angles, similarities)  
        plt.fill(angles, similarities, alpha=0.25)  

    def _compute_physics_similarity(self, layer_vis):  
        """计算物理相似度"""  
        return np.corrcoef(  
            layer_vis['weight_mu'].flatten(),   
            layer_vis['physics_kernel'].flatten()  
        )[0, 1]      
    def get_physics_interpretation(self):
        """获取物理解释信息"""
        interpretations = []
        
        for i, vk_layer in enumerate(self.vk_layers):
            layer_info = {
                'layer_idx': i,
                'kernel_size': vk_layer.kernel_size,
                'weight_mean': vk_layer.weight_mu.detach().cpu().numpy(),
                'weight_std': torch.exp(vk_layer.weight_log_sigma).detach().cpu().numpy(),
                'physics_similarity': torch.cosine_similarity(
                    vk_layer.weight_mu.flatten(),
                    vk_layer.physics_kernel.flatten(),
                    dim=0
                ).item()
            }
            interpretations.append(layer_info)
        
        return interpretations


# 修改模块测试代码  
if __name__ == "__main__":
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from configs.config import config
    from data.university_of_ottawa_loader import UniversityOfOttawaDataLoader
    from preprocess.data_preprocessor import DataPreprocessor
    from models.pi_vkcnn import PI_VKCNN
    import seaborn as sns
    import sys
    sys.path.append(r'E:\20250711电机小论文')

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

    # 创建模型
    model = PI_VKCNN(config)

    # 准备测试数据
    def prepare_test_data(signals, labels, num_samples=32):
        """准备测试数据"""
        indices = np.random.choice(len(signals), num_samples, replace=False)
        test_signals = [signals[i] for i in indices]
        test_labels = [labels[i] for i in indices]
        
        # 转换为张量
        test_signals = torch.tensor(test_signals, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
        
        return test_signals, test_labels

    # 选择振动信号1作为测试数据
    test_input, test_labels = prepare_test_data(processed_vibration_1, processed_labels)

    print(f"\n测试PI_VKCNN模块:")
    print(f"输入形状: {test_input.shape}")
    print(f"标签形状: {test_labels.shape}")

    # 前向传播
    with torch.no_grad():
        output, losses = model(test_input)
        
        print(f"输出形状: {output.shape}")
        print(f"KL损失: {losses['kl_loss']:.4f}")
        print(f"物理约束损失: {losses['physics_loss']:.4f}")
        print(f"因果性损失: {losses['causality_loss']:.4f}")
        
        # 获取物理解释
        interpretations = model.get_physics_interpretation()
        print(f"\n物理解释信息:")
        for i, interp in enumerate(interpretations):
            print(f"层 {i}: 物理相似度 = {interp['physics_similarity']:.4f}")
        
        # 可视化物理解释
        model.visualize_physics_interpretation(
            save_path=r'E:\20250711电机小论文\results\pi_vkcnn_interpretation'
        )

        # 额外的可视化：特征分布
        plt.figure(figsize=(10, 6))
        plt.title("模型输出特征分布")
        for label in torch.unique(test_labels):
            mask = test_labels == label
            features_subset = output[mask]
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
        plt.savefig(r'E:\20250711电机小论文\results\pi_vkcnn_feature_distribution.png')
        plt.close()

    print(f"\nPI_VKCNN模块测试完成!")
