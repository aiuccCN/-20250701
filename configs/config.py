class Config:
    """University of Ottawa数据集实验配置"""
    # 数据集参数
    SAMPLE_RATE = 42000  # University of Ottawa数据集采样率
    SIGNAL_LENGTH = 8192  # 信号长度
    NUM_CLASSES = 8  # 故障类别数

    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 10
    PATIENCE = 5  # 早停耐心值
    FEATURE_DIM = 256
    
    # 数据预处理参数
    OVERLAP_RATIO = 0.5  # 窗口重叠比例
    WINDOW_SIZE = 2048  # 时间窗口大小

    # PI-VKCNN参数
    PI_VKCNN_KERNELS = [32, 16, 8, 4]  # 卷积核大小
    PI_VKCNN_CHANNELS = [1, 32, 64, 128, 256]  # 通道数

    # MLWCN参数
    MLWCN_SCALES = 5  # 小波分解层数
    MLWCN_WAVELET = 'db4'  # 小波基函数
    WAVELET_TYPES = ['db4', 'db8', 'haar', 'coif2', 'bior2.2']  # 支持的小波类型

    # CAAF参数
    CAAF_HEADS = 8  # 注意力头数
    CAAF_HIDDEN_DIM = 256  # 隐藏维度
    CAAF_DROPOUT = 0.5  # dropout概率

    # 物理约束参数
    ALPHA = 0.05  # 物理约束权重
    BETA = 0.01  # 稀疏性约束权重
    GAMMA = 0.02  # 一致性约束权重
    LAMBDA_SPARSE = 0.001  # 稀疏正则化参数
    CAUSALITY_THRESHOLD = 0.05  # 因果约束阈值

    # 故障类别标签
    FAULT_LABELS = [
        'Normal', 'Rotor_Unbalance', 'Rotor_Misalignment', 
        'Bearing_Fault', 'Stator_Fault', 'Voltage_Unbalance',
        'Bent_Rotor', 'Broken_Rotor_Bar'
    ]

    # 数据增强参数
    AUGMENT_PROB = 0.5  # 数据增强概率
    NOISE_LEVELS = [0.01, 0.02, 0.03, 0.05]  # 噪声水平
    AMPLITUDE_RANGE = (0.8, 1.2)  # 幅值缩放范围
    TIME_SHIFT_RANGE = (-512, 512)  # 时间偏移范围

# 创建配置实例
config = Config()
