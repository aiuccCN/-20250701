import os
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import sys
sys.path.append(r'E:\20250711电机小论文')

class UniversityOfOttawaDataLoader:
    """University of Ottawa数据集加载器"""
    def __init__(self, data_root, config):
        self.data_root = data_root
        self.config = config
        
        # 故障类型映射
        self.fault_mapping = {
            'H_H': 0,  # Normal (Healthy)
            'R_U': 1,  # Rotor Unbalance
            'R_M': 2,  # Rotor Misalignment
            'F_B': 3,  # Bearing Fault
            'S_W': 4,  # Stator Winding Fault
            'V_U': 5,  # Voltage Unbalance
            'B_R': 6,  # Broken Rotor Bar
            'K_A': 7   # Other Fault
        }
        
        # 故障类型名称
        self.fault_names = {
            0: 'Normal',
            1: 'Rotor_Unbalance', 
            2: 'Rotor_Misalignment',
            3: 'Bearing_Fault',
            4: 'Stator_Fault',
            5: 'Voltage_Unbalance',
            6: 'Broken_Rotor_Bar',
            7: 'Other_Fault'
        }
        
        print("University of Ottawa数据集加载器初始化完成")
        print(f"数据根目录: {data_root}")
        print(f"故障类型映射: {self.fault_mapping}")

    def load_csv_data(self, file_path):
        """加载CSV数据文件"""
        try:
            data = pd.read_csv(file_path)
            return data.values
        except Exception as e:
            print(f"加载CSV文件 {file_path} 失败: {e}")
            return None

    def load_mat_data(self, file_path):
        """加载MAT数据文件"""
        try:
            data = loadmat(file_path)
            # 根据实际MAT文件结构调整键名
            keys = [k for k in data.keys() if not k.startswith('__')]
            if len(keys) > 0:
                return data[keys[0]]
            return None
        except Exception as e:
            print(f"加载MAT文件 {file_path} 失败: {e}")
            return None

    def parse_filename(self, filename):
        """解析文件名获取故障类型和工况信息"""
        # 文件名格式: 故障类型_编号_工况.csv/mat
        basename = os.path.basename(filename)
        name_parts = basename.split('_')
        
        if len(name_parts) >= 3:
            fault_type = f"{name_parts[0]}_{name_parts[1]}"  # 如 R_U, F_B
            sensor_id = name_parts[2]
            condition = name_parts[3].split('.')[0]  # 0为空载，1为负载
            
            return fault_type, int(sensor_id), int(condition)
        
        return None, None, None

    def load_dataset(self, use_csv=True, conditions=[0, 1]):
        """
        加载完整数据集
        
        Args:
            use_csv: 是否使用CSV文件（否则使用MAT文件）
            conditions: 加载的工况条件 [0: 空载, 1: 负载]
        """
        print("开始加载University of Ottawa数据集...")
        
        # 选择数据格式
        if use_csv:
            data_dir = os.path.join(self.data_root, "2_CSV_Data_Files")
            file_ext = "*.csv"
        else:
            data_dir = os.path.join(self.data_root, "3_MatLab_Data_Files")
            file_ext = "*.mat"
        
        all_data = []
        all_labels = []
        all_metadata = []
        
        # 遍历工况条件
        for condition in conditions:
            if condition == 0:
                condition_dir = os.path.join(data_dir, "1_Unloaded_Condition")
                condition_name = "空载"
            else:
                condition_dir = os.path.join(data_dir, "2_Loaded_Condition")
                condition_name = "负载"
            
            print(f"\n加载{condition_name}工况数据...")
            
            # 获取所有数据文件
            files = glob.glob(os.path.join(condition_dir, file_ext))
            files.sort()
            
            condition_count = 0
            for file_path in files:
                fault_type, sensor_id, file_condition = self.parse_filename(file_path)
                
                if fault_type not in self.fault_mapping:
                    print(f"未知故障类型: {fault_type}")
                    continue
                
                # 加载数据
                if use_csv:
                    data = self.load_csv_data(file_path)
                else:
                    data = self.load_mat_data(file_path)
                
                if data is not None:
                    # 正确提取信号
                    
                    accelerometer_1_signal = data[:, 0]  # 第1列，振动信号
                    microphone_signal = data[:, 1]  # 第2列，音频信号
                    accelerometer_2_signal = data[:, 2]  # 第3列 振动信号 
                    accelerometer_3_signal = data[:, 3]  # 第4列 振动信号 
                    temperature_signal = data[:, 4]  # 第5列，温度
                    
                    # 获取标签
                    label = self.fault_mapping[fault_type]
                    
                    # 存储数据和元信息
                    all_data.append({
                        'accelerometer_1': accelerometer_1_signal,
                        'microphone': microphone_signal,
                        'accelerometer_2': accelerometer_2_signal,
                        'accelerometer_3': accelerometer_3_signal,
                        'temperature': temperature_signal
                    })
                    all_labels.append(label)
                    all_metadata.append({
                        'fault_type': fault_type,
                        'fault_name': self.fault_names[label],
                        'sensor_id': sensor_id,
                        'condition': condition,
                        'condition_name': condition_name,
                        'file_path': file_path
                    })
                    
                    condition_count += 1
            
            print(f"{condition_name}工况加载完成: {condition_count} 个文件")
        
        print(f"\n数据集加载完成!")
        print(f"总样本数: {len(all_data)}")
        print(f"故障类型分布:")
        label_counts = {}
        for label in all_labels:
            fault_name = self.fault_names[label]
            label_counts[fault_name] = label_counts.get(fault_name, 0) + 1
        
        for fault_name, count in label_counts.items():
            print(f"  {fault_name}: {count} 个样本")
        
        return all_data, all_labels, all_metadata

# 测试数据加载器
if __name__ == "__main__":
    from configs.config import config
    
    # 数据集根目录 - 请根据您的实际路径修改
    DATA_ROOT = "E:/20250711电机小论文/data"
    
    # 初始化数据加载器
    data_loader = UniversityOfOttawaDataLoader(DATA_ROOT, config)
    
    # 加载数据集
    try:
        data_list, labels, metadata = data_loader.load_dataset(
            use_csv=True,  # 使用CSV文件
            conditions=[0, 1]  # 加载空载和负载工况
        )
        
        print(f"\n数据加载统计:")
        print(f"数据样本数: {len(data_list)}")
        print(f"标签数量: {len(labels)}")
        print(f"元数据数量: {len(metadata)}")
        
        # 显示前几个样本的详细信息
        print("\n前3个样本详细信息:")
        for i in range(min(3, len(data_list))):
            print(f"\n样本 {i + 1}:")
            print(f"  加速度计1信号长度: {len(data_list[i]['accelerometer_1'])}")
            print(f"  麦克风信号长度: {len(data_list[i]['microphone'])}")
            print(f"  加速度计2信号长度: {len(data_list[i]['accelerometer_2'])}")
            print(f"  加速度计3信号长度: {len(data_list[i]['accelerometer_3'])}")
            print(f"  温度信号长度: {len(data_list[i]['temperature'])}")
            print(f"  标签: {labels[i]} ({data_loader.fault_names[labels[i]]})")
            print(f"  元数据: {metadata[i]}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请检查数据目录路径和文件格式")
