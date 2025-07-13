import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import sys
sys.path.append(r'E:\20250711电机小论文')
from torch.utils.data import Dataset  
from models.pi_vkcnn import PI_VKCNN
from models.mlwcn import MLWCN
class MotorFaultDataset(Dataset):
    def __init__(self, acoustic_data, vibration_data, labels):
        self.acoustic_data = torch.FloatTensor(acoustic_data)
        self.labels = torch.LongTensor(labels)
        
        # 分别处理每种传感器数据
        self.vibration_data = {}
        for sensor_name, data in vibration_data.items():
            self.vibration_data[sensor_name] = torch.FloatTensor(data)
            
        # 验证所有传感器数据的样本数一致
        self.num_samples = len(self.acoustic_data)
        for sensor_data in self.vibration_data.values():
            assert len(sensor_data) == self.num_samples, f"传感器数据样本数不一致: {len(sensor_data)} != {self.num_samples}"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回一个样本的所有模态数据
        sample = {
            'acoustic': self.acoustic_data[idx],
            'vibration': {sensor: data[idx] for sensor, data in self.vibration_data.items()},
            'label': self.labels[idx]
        }
        return sample

class IntegratedFaultDiagnosisModel(nn.Module):  
    """集成故障诊断模型"""  
    def __init__(self, config):  
        super(IntegratedFaultDiagnosisModel, self).__init__()  
        self.config = config  
        
        # 使用绝对导入  
        from models.pi_vkcnn import PI_VKCNN  
        from models.mlwcn import MLWCN  
        
        self.pi_vkcnn = PI_VKCNN(config)  
        self.mlwcn = MLWCN(config)  
        
        # 特征融合层  
        self.feature_fusion = nn.Sequential(  
            nn.Linear(512, 256),  # 256 + 256 = 512  
            nn.ReLU(),  
            nn.Dropout(0.3),  
            nn.Linear(256, 128),  
            nn.ReLU(),  
            nn.Dropout(0.3)  
        )  
        
        # 分类器  
        self.classifier = nn.Sequential(  
            nn.Linear(128, 64),  
            nn.ReLU(),  
            nn.Dropout(0.3),  
            nn.Linear(64, config.NUM_CLASSES)  
        )  
        
        print(f"IntegratedFaultDiagnosisModel初始化完成")  

    def forward(self, acoustic_data, vibration_data, labels=None):
        # 准备5通道输入信号
        input_signals = torch.stack([
            acoustic_data['microphone'],      # 声学信号
            vibration_data['accelerometer_1'], # 主振动信号
            vibration_data['accelerometer_2'], # 次振动信号1
            vibration_data['accelerometer_3'], # 次振动信号2
            vibration_data['temperature']      # 温度信号
        ], dim=1)

        # PI_VKCNN处理多通道信号  
        pi_features, pi_losses = self.pi_vkcnn(input_signals)  
        
        # MLWCN处理多通道信号  
        mlwcn_features, mlwcn_losses = self.mlwcn(input_signals, labels)  
        
        # 特征融合  
        fused_features = torch.cat([pi_features, mlwcn_features], dim=1)  
        features = self.feature_fusion(fused_features)  
        
        # 分类  
        logits = self.classifier(features)
        
        # 合并所有损失  
        total_losses = {}  
        for key, value in pi_losses.items():  
            total_losses[f'pi_{key}'] = value  
        for key, value in mlwcn_losses.items():  
            total_losses[f'mlwcn_{key}'] = value  
        
        return logits, total_losses

class CrossModalAdaptiveAttentionFusion(nn.Module):     
    # 跨模态注意力
    def __init__(self, config):
        super().__init__()
        
        # 多头注意力机制 - 增强版
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=config.FEATURE_DIM,
            num_heads=config.CAAF_HEADS,
            dropout=config.CAAF_DROPOUT
        )
        
        # 自适应掩码层 - 概率增强
        self.adaptive_mask = nn.Sequential(
            nn.Linear(config.FEATURE_DIM * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.FEATURE_DIM),
            nn.Sigmoid()
        )
        
        # 不确定性感知融合 - 概率建模
        self.uncertainty_fusion = nn.Sequential(
            nn.Linear(config.FEATURE_DIM * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.CAAF_DROPOUT),
            nn.Linear(512, config.FEATURE_DIM)
        )
        
        # 对比学习温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, vibration_features, acoustic_features, labels=None):
        # 跨模态注意力 - 增强版
        attn_output, attn_weights = self.cross_modal_attention(
            vibration_features.unsqueeze(0),   
            acoustic_features.unsqueeze(0),   
            acoustic_features.unsqueeze(0)
        )
        
        # 上下文特征增强
        context_features = torch.cat([
            vibration_features, 
            attn_output.squeeze(0), 
            acoustic_features
        ], dim=-1)
        
        # 自适应掩码 - 概率增强
        adaptive_mask = self.adaptive_mask(context_features)
        
        # 特征加权 - 概率融合
        fused_features = (
            vibration_features * adaptive_mask + 
            attn_output.squeeze(0) * (1 - adaptive_mask)
        )
        
        # 不确定性感知融合
        final_features = self.uncertainty_fusion(
            torch.cat([vibration_features, fused_features], dim=1)
        )
        
        # 损失计算 - 对比学习增强
        losses = {}
        if labels is not None:
            losses['modal_alignment_loss'] = self._compute_enhanced_alignment_loss(
                vibration_features, 
                acoustic_features, 
                labels
            )
        
        return final_features, losses
    
    def _compute_enhanced_alignment_loss(self, vibration_features, acoustic_features, labels):
        """增强版模态对齐损失"""
        # 余弦相似度
        similarity = F.cosine_similarity(vibration_features, acoustic_features, dim=-1)
        
        # 同类别和不同类别掩码
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        neg_mask = 1 - pos_mask
        
        # 带温度的对比损失
        pos_loss = torch.mean(1 - similarity * pos_mask / self.temperature)
        neg_loss = torch.mean(torch.clamp(similarity * neg_mask / self.temperature, min=0))
        
        return pos_loss + neg_loss


class InterpretableMotorFaultDiagnosisModel(nn.Module):
    """可解释性电机故障诊断模型"""
    def __init__(self, config):
        super().__init__()
        
        # 核心模块
        self.pi_vkcnn = PI_VKCNN(config)
        self.mlwcn = MLWCN(config)
        self.caaf = CrossModalAdaptiveAttentionFusion(config)
        
        # 可解释性决策层
        self.decision_layer = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.NUM_CLASSES)
        )
        
        # 物理可解释性模块
        self.physics_interpretation_module = PhysicsInterpretationModule(config)
    
    def forward(self, vibration_signals, acoustic_signals, labels=None):
        """
        端到端的可解释性故障诊断前向传播
        
        参数:
        - vibration_signals: 振动信号 [batch_size, 5, signal_length]
        - acoustic_signals: 声学信号 [batch_size, 5, signal_length]
        - labels: 标签（可选）
        """
        # 处理振动信号，如果是字典则转换为张量
        if isinstance(vibration_signals, dict):
            vibration_signals = torch.stack([
                vibration_signals['accelerometer_1'],
                vibration_signals['microphone'],
                vibration_signals['accelerometer_2'],
                vibration_signals['accelerometer_3'],
                vibration_signals['temperature']
            ], dim=1)
        
        # 处理声学信号，如果是字典则转换为张量
        if isinstance(acoustic_signals, dict):
            acoustic_signals = torch.stack([
                acoustic_signals['accelerometer_1'],
                acoustic_signals['microphone'],
                acoustic_signals['accelerometer_2'],
                acoustic_signals['accelerometer_3'],
                acoustic_signals['temperature']
            ], dim=1)
        
        # 1. 物理约束特征提取
        pi_features, pi_losses = self.pi_vkcnn(vibration_signals)
        
        # 2. 对比学习特征提取
        mlwcn_features, mlwcn_losses = self.mlwcn(acoustic_signals, labels)
        
        # 3. 跨模态自适应融合
        fused_features, caaf_losses = self.caaf(
            pi_features, 
            mlwcn_features, 
            labels
        )
        
        # 4. 故障诊断分类
        classification_logits = self.decision_layer(fused_features)
        
        # 5. 损失汇总
        total_losses = {
            **pi_losses,
            **mlwcn_losses,
            **caaf_losses,
            'classification_loss': F.cross_entropy(classification_logits, labels)
        }
        
        # 6. 物理可解释性
        physics_explanation = self.physics_interpretation_module(
            pi_features, 
            mlwcn_features, 
            classification_logits
        )
        
        return classification_logits, total_losses, physics_explanation

class PhysicsInterpretationModule(nn.Module):
    """物理可解释性模块"""
    def __init__(self, config):
        super().__init__()
        
        # 物理约束特征解释
        self.physics_feature_explainer = nn.Sequential(
            nn.Linear(config.FEATURE_DIM * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 故障机理映射
        self.fault_mechanism_mapper = nn.Sequential(
            nn.Linear(config.NUM_CLASSES, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, pi_features, mlwcn_features, classification_logits):
        """
        解析物理特征和故障机理
        
        返回:
        - feature_explanation: 特征物理解释
        - mechanism_explanation: 故障机理解释
        """
        # 特征物理解释
        feature_explanation = self.physics_feature_explainer(
            torch.cat([pi_features, mlwcn_features], dim=1)
        )
        
        # 故障机理映射
        mechanism_explanation = self.fault_mechanism_mapper(
            F.softmax(classification_logits, dim=1)
        )
        
        return {
            'feature_explanation': feature_explanation,
            'mechanism_explanation': mechanism_explanation
        }
