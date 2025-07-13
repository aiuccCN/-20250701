import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import seaborn as sns
import sys
sys.path.append(r'E:\20250711电机小论文')
from utils.early_stopping import EarlyStoppingCallback
from configs.config import config
from data.university_of_ottawa_loader import UniversityOfOttawaDataLoader
from preprocess.data_preprocessor import DataPreprocessor
from models.integrated_model import InterpretableMotorFaultDiagnosisModel, MotorFaultDataset
from utils.training import train_model, visualize_training_process


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


def main():
    device = setup_test_environment()
    train_loader, val_loader = load_and_preprocess_data()

    model = InterpretableMotorFaultDiagnosisModel(config).to(device)
    train_losses, val_accuracies, best_val_acc = train_model(model, train_loader, val_loader, config, device)

    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")

    visualize_training_process(train_losses, val_accuracies)


if __name__ == "__main__":
    main()