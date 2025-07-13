import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import sys
sys.path.append(r'E:\20250711电机小论文')
from utils.early_stopping import EarlyStoppingCallback
from configs.config import config, Config

def train_model(model, train_loader, val_loader, config, device):
    """
    高级模型训练函数，支持物理可解释性约束的端到端训练
    
    Args:
        model (nn.Module): 可解释性故障诊断模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        config (Config): 配置对象
        device (torch.device): 计算设备
    
    Returns:
        tuple: 训练损失、验证准确率和最佳验证准确率
    """
    # 优化器配置：Adam + 权重衰减
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=1e-4,  # L2正则化
        betas=(0.9, 0.999)  # 自适应矩估计
    )

    # 学习率调度器：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 初始重启周期
        T_mult=2,  # 重启周期倍增
        eta_min=1e-5  # 最小学习率
    )

    # 早停机制
    early_stopping = EarlyStoppingCallback(
        patience=config.PATIENCE, 
        min_delta=0.001, 
        mode='max',
        physics_tolerance=0.05
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑

    # 梯度累积步数
    accumulation_steps = 4  # 可以根据需要调整

    # 性能记录
    train_losses, val_accuracies = [], []
    best_val_acc = 0.0

    # 训练进度条信息
    print(f"\n开始训练可解释性故障诊断模型...")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")

    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据准备
            acoustic_data = batch['acoustic'].to(device)
            vibration_data = {
                'accelerometer_1': batch['vibration']['accelerometer_1'].to(device),
                'microphone': batch['vibration']['microphone'].to(device),
                'accelerometer_2': batch['vibration']['accelerometer_2'].to(device),
                'accelerometer_3': batch['vibration']['accelerometer_3'].to(device),
                'temperature': batch['vibration']['temperature'].to(device)
            }
            labels = batch['label'].to(device)
            
            # 前向传播
            logits, losses, physics_explanation = model(acoustic_data, vibration_data, labels)
            
            # 分类损失
            classification_loss = criterion(logits, labels)
            
            # 物理约束损失
            physics_losses = sum(losses.values())
            
            # 总损失
            total_loss = classification_loss + physics_losses
            
            # 反向传播
            total_loss = total_loss / accumulation_steps  # 缩放损失以进行梯度累积
            total_loss.backward()
            
            # 性能统计
            train_loss += total_loss.item() * accumulation_steps  # 恢复损失尺度以进行统计
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 训练进度打印
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], '
                      f'Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {total_loss.item() * accumulation_steps:.4f}, '
                      f'Acc: {100.0 * train_correct / train_total:.2f}%')
            
            # 梯度累积和参数更新
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 参数更新
                optimizer.step()
                
                # 梯度清零
                optimizer.zero_grad()
        
        # 如果最后一个批次没有达到累积步数，仍需更新参数
        if batch_idx % accumulation_steps != 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            
            # 梯度清零
            optimizer.zero_grad()
        
        # 学习率调度
        scheduler.step(epoch)
        
        # 验证阶段
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                acoustic_data = batch['acoustic'].to(device)
                vibration_data = {
                    'accelerometer_1': batch['vibration']['accelerometer_1'].to(device),
                    'microphone': batch['vibration']['microphone'].to(device),
                    'accelerometer_2': batch['vibration']['accelerometer_2'].to(device),
                    'accelerometer_3': batch['vibration']['accelerometer_3'].to(device),
                    'temperature': batch['vibration']['temperature'].to(device)
                }
                labels = batch['label'].to(device)
                
                # 前向传播
                logits, _, _ = model(acoustic_data, vibration_data, labels)
                
                # 性能统计
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 计算指标
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        # 性能打印
        print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}]: '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')
        
        # 模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'新的最佳模型已保存 (Val Acc: {val_acc:.2f}%)')
        
        # 早停检查
        physics_similarity = np.mean([losses.get('physics_similarity', 0) for losses in [losses]])
        if early_stopping(val_acc, physics_similarity):
            print(f'早停触发，停止训练')
            break
    
    return train_losses, val_accuracies, best_val_acc

def visualize_training_process(train_losses, val_accuracies):
    """
    可视化训练过程
    
    Args:
        train_losses (list): 训练损失
        val_accuracies (list): 验证准确率
    """
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('训练损失')
    plt.plot(train_losses)
    plt.xlabel('轮次')
    plt.ylabel('损失')
    
    plt.subplot(1, 3, 2)
    plt.title('验证准确率')
    plt.plot(val_accuracies)
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    
    plt.subplot(1, 3, 3)
    plt.title('学习率变化')
    plt.plot(np.linspace(config.LEARNING_RATE, 1e-5, len(train_losses)))
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    
    plt.tight_layout()
    plt.savefig('training_process.png')
    plt.show()

if __name__ == "__main__":
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from configs.config import config
    from data.university_of_ottawa_loader import UniversityOfOttawaDataLoader
    from preprocess.data_preprocessor import DataPreprocessor
    from models.integrated_model import InterpretableMotorFaultDiagnosisModel, MotorFaultDataset

    def setup_test_environment():
        """设置测试环境"""
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

        # 检查设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"测试设备: {device}")
        return device

    def load_and_preprocess_data():
        """加载和预处理数据"""
        # 数据集根目录
        DATA_ROOT = "./data"

        # 初始化数据加载器
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
            data_list, labels, metadata, apply_augmentation=True
        )

        # 数据划分 - 更清晰的变量命名和数据组织
        X_acoustic = np.array(processed_microphone)
        X_accelerometer_1 = np.array(processed_vibration_1)
        X_accelerometer_2 = np.array(processed_vibration_2)
        X_accelerometer_3 = np.array(processed_vibration_3)
        X_temperature = np.array(processed_temperature)
        y = np.array(processed_labels)

        # 训练集和验证集划分 - 正确处理多个输入数组
        split_results = train_test_split(
            X_acoustic,  # 声学数据
            X_accelerometer_1,  # 加速度计1
            X_accelerometer_2,  # 加速度计2
            X_accelerometer_3,  # 加速度计3
            X_temperature,  # 温度数据
            y,  # 标签
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )

        # 解包分割结果
        X_train_acoustic, X_val_acoustic, \
        X_train_accelerometer_1, X_val_accelerometer_1, \
        X_train_accelerometer_2, X_val_accelerometer_2, \
        X_train_accelerometer_3, X_val_accelerometer_3, \
        X_train_temperature, X_val_temperature, \
        y_train, y_val = split_results

        # 创建数据集和数据加载器
        train_dataset = MotorFaultDataset(
            acoustic_data=X_train_acoustic, 
            vibration_data={
                'accelerometer_1': X_train_accelerometer_1,
                'microphone': X_train_acoustic,  # 声学数据实际上是麦克风数据
                'accelerometer_2': X_train_accelerometer_2,
                'accelerometer_3': X_train_accelerometer_3,
                'temperature': X_train_temperature
            }, 
            labels=y_train
        )

        val_dataset = MotorFaultDataset(
            acoustic_data=X_val_acoustic, 
            vibration_data={
                'accelerometer_1': X_val_accelerometer_1,
                'microphone': X_val_acoustic,  # 声学数据实际上是麦克风数据
                'accelerometer_2': X_val_accelerometer_2,
                'accelerometer_3': X_val_accelerometer_3,
                'temperature': X_val_temperature
            }, 
            labels=y_val
        )

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        return train_loader, val_loader

    device = setup_test_environment()
    train_loader, val_loader = load_and_preprocess_data()

    model = InterpretableMotorFaultDiagnosisModel(config).to(device)
    train_losses, val_accuracies, best_val_acc = train_model(model, train_loader, val_loader, config, device)

    visualize_training_process(train_losses, val_accuracies)