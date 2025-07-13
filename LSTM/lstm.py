import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from typing import List, Dict, Tuple, Optional

# 设置随机种子以确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

def parse_label(label: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    解析样本标记，返回(故障类型1, 故障类型2, 频率代码, 负载代码)
    如果标记格式不正确，返回(None, None, None, None)
    """
    parts = label.strip().split('-')
    if len(parts) != 4:
        return None, None, None, None
    
    fault_type1, fault_type2, freq_code, load_code = parts
    # 简单验证各部分是否符合预期格式
    if (fault_type1 not in {'H', 'R', 'S', 'V', 'B', 'K', 'F'} or
        fault_type2 not in {'H', 'U', 'M', 'W', 'R', 'A', 'B'} or
        freq_code not in {'1', '2', '3', '4', '5', '6', '7'} or
        load_code not in {'0', '1'}):
        return None, None, None, None
    
    return fault_type1, fault_type2, freq_code, load_code

def get_health_description(fault_type1: str, fault_type2: str) -> str:
    """根据故障类型代码获取健康状况描述"""
    fault1_desc = {
        'H': '正常', 'R': '转子', 'S': '定子', 
        'V': '电压', 'B': '弯曲', 'K': '断裂', 'F': '故障'
    }
    fault2_desc = {
        'H': '正常', 'U': '不平衡', 'M': '不对中', 
        'W': '绕组', 'R': '转子', 'A': '转子条', 'B': '轴承'
    }
    
    desc1 = fault1_desc.get(fault_type1, '未知')
    desc2 = fault2_desc.get(fault_type2, '未知')
    
    if fault_type1 == 'H' and fault_type2 == 'H':
        return "正常运行"
    elif fault_type1 == 'H':
        return f"{desc2}故障"
    elif fault_type2 == 'H':
        return f"{desc1}故障"
    else:
        return f"{desc1}{desc2}故障"

def load_data(directory: str) -> pd.DataFrame:
    """
    加载指定目录下的所有CSV文件并合并
    """
    all_dfs = []
    total_files = 0
    
    # 遍历目录中的所有文件
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            total_files += 1
            file_path = os.path.join(directory, file)
            
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 从文件名中提取标签（假设格式为"标签.csv"）
                file_name = os.path.splitext(file)[0]
                # 将文件名中的下划线转换为连字符以匹配标记格式
                file_label = file_name.replace('_', '-')
                
                # 添加标签列
                df['label'] = file_label
                
                all_dfs.append(df)
                
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
    
    if not all_dfs:
        print(f"警告: 在目录 {directory} 中未找到CSV文件")
        return pd.DataFrame()
    
    # 合并所有数据框
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"数据加载完成，共处理 {total_files} 个CSV文件")
    print(f"合并后的数据包含 {len(combined_df)} 条记录")
    
    return combined_df

def prepare_data(df: pd.DataFrame, label_column: str = 'label', 
                 sequence_length: int = 50, test_size: float = 0.2, 
                 val_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    准备LSTM模型的训练数据
    """
    # 提取特征列（假设除标签列外的所有列都是特征）
    feature_columns = [col for col in df.columns if col != label_column]
    
    # 提取特征和标签
    X = df[feature_columns].values
    y = df[label_column].apply(lambda x: parse_label(x)[0] + '-' + parse_label(x)[1]).values
    
    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 创建序列数据
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y_encoded[i+sequence_length-1])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # 划分训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, random_state=42, stratify=y_seq
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=42, stratify=y_train_val
    )
    
    # 标准化特征
    scaler = StandardScaler()
    n_samples, seq_len, n_features = X_train.shape
    
    X_train_reshaped = X_train.reshape(n_samples * seq_len, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train = X_train_scaled.reshape(n_samples, seq_len, n_features)
    
    X_val_reshaped = X_val.reshape(X_val.shape[0] * seq_len, n_features)
    X_val = scaler.transform(X_val_reshaped).reshape(X_val.shape[0], seq_len, n_features)
    
    X_test_reshaped = X_test.reshape(X_test.shape[0] * seq_len, n_features)
    X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape[0], seq_len, n_features)
    
    print(f"数据准备完成:")
    print(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}, 测试集形状: {X_test.shape}")
    print(f"类别分布: {dict(zip(le.classes_, np.bincount(y_encoded)))}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le

def build_lstm_model(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """
    构建LSTM模型
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                input_shape: Tuple[int, int], num_classes: int, 
                epochs: int = 50, batch_size: int = 32) -> Tuple[tf.keras.Model, Dict]:
    """
    训练LSTM模型，增强早停机制
    """
    # 创建模型
    model = build_lstm_model(input_shape, num_classes)
    
    # 设置早停回调（增强版）
    early_stopping = EarlyStopping(
        monitor='val_loss',      # 监控验证集损失
        patience=3,             # 15轮没有改善则停止
        restore_best_weights=True,  # 恢复最佳权重
        verbose=1,               # 显示早停信息
        min_delta=0.001         # 最小改善阈值
    )
    
    # 模型检查点回调
    model_checkpoint = ModelCheckpoint(
        'best_lstm_model.h5',    # 保存最佳模型
        monitor='val_accuracy',  # 监控验证集准确率
        save_best_only=True,     # 只保存最佳模型
        mode='max',              # 准确率越高越好
        verbose=1                # 显示保存信息
    )
    
    # 学习率调度回调
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',      # 监控验证集损失
        factor=0.3,              # 学习率降低因子
        patience=2,              # 5轮没有改善则降低学习率
        min_lr=0.00001,          # 最小学习率
        verbose=1                # 显示学习率调整信息
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler],
        verbose=1
    )
    
    return model, history.history

def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, 
                  label_encoder: LabelEncoder, output_dir: str) -> None:
    """
    评估模型性能并保存结果
    """
    # 创建结果输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n测试集准确率: {test_acc:.4f}, 测试集损失: {test_loss:.4f}")
    
    # 保存评估结果到文件
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"测试集准确率: {test_acc:.4f}\n")
        f.write(f"测试集损失: {test_loss:.4f}\n")
    
    # 预测并生成分类报告
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("\n分类报告:")
    print(report)
    
    # 保存分类报告到文件
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    
    # 在混淆矩阵上标注数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"\n混淆矩阵已保存到: {os.path.join(output_dir, 'confusion_matrix.png')}")
    
    # 保存预测结果到CSV文件
    predictions_df = pd.DataFrame({
        'true_label': label_encoder.inverse_transform(y_test),
        'predicted_label': label_encoder.inverse_transform(y_pred),
        'is_correct': y_test == y_pred
    })
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"预测结果已保存到: {os.path.join(output_dir, 'predictions.csv')}")

def visualize_training_history(history: Dict, output_dir: str) -> None:
    """
    可视化训练历史并保存图表
    """
    # 创建结果输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('训练轮次')
    plt.legend(['训练', '验证'], loc='lower right')
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('训练轮次')
    plt.legend(['训练', '验证'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    print(f"\n训练历史图表已保存到: {os.path.join(output_dir, 'training_history.png')}")

def main():
    # 配置参数
    data_directory = r'E:\20250711电机小论文\data\2_CSV_Data_Files\15hz无负载'  # 数据目录路径
    output_directory = r'E:\20250711电机小论文\LSTM\results'  # 结果输出目录
    label_column = 'label'  # 标签列名
    sequence_length = 5  # LSTM序列长度
    test_size = 0.2  # 测试集比例
    val_size = 0.1  # 验证集比例
    epochs = 40  # 训练轮次（设置较大值，依赖早停机制）
    batch_size = 64  # 批次大小
    
    # 检查数据目录是否存在
    if not os.path.exists(data_directory):
        print(f"错误: 数据目录 '{data_directory}' 不存在!")
        return
    
    # 加载数据
    print(f"开始从 {data_directory} 加载数据...")
    df = load_data(data_directory)
    if df.empty:
        return
    
    # 准备数据
    print("\n开始准备LSTM模型训练数据...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = prepare_data(
        df, label_column, sequence_length, test_size, val_size
    )
    
    # 训练模型
    print("\n开始训练LSTM模型...")
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        (sequence_length, X_train.shape[2]),
        len(label_encoder.classes_),
        epochs, batch_size
    )
    
    # 评估模型并保存结果
    print("\n开始评估模型并保存结果...")
    evaluate_model(model, X_test, y_test, label_encoder, output_directory)
    
    # 可视化训练历史并保存图表
    visualize_training_history(history, output_directory)
    
    print(f"\nLSTM模型训练和评估完成！所有结果已保存到: {output_directory}")

if __name__ == "__main__":
    main()