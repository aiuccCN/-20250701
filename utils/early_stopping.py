import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import sys
sys.path.append(r'E:\20250711电机小论文')

class EarlyStoppingCallback:
    """
    针对可解释性电机故障诊断的智能早停回调
    
    特点:
    1. 多指标监控
    2. 物理可解释性评估
    3. 动态调整容忍度
    4. 性能趋势分析
    """
    def __init__(
        self, 
        patience=20, 
        min_delta=0.001, 
        mode='max', 
        performance_window=5,
        physics_tolerance=0.05
    ):
        """
        初始化早停类
        
        Args:
            patience (int): 容忍轮数
            min_delta (float): 性能改善最小阈值
            mode (str): 性能指标模式
            performance_window (int): 性能趋势分析窗口
            physics_tolerance (float): 物理可解释性容忍度
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.performance_window = performance_window
        self.physics_tolerance = physics_tolerance
        
        # 性能相关变量
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        # 性能历史记录
        self.performance_history = []
        self.physics_history = []
        
        # 动态调整参数
        self.adaptive_patience = patience
        
        # 性能趋势评估
        self.compare_func = (
            lambda current, best: current > best + self.min_delta 
            if self.mode == 'max' 
            else current < best - self.min_delta
        )
    
    def __call__(
        self, 
        current_score, 
        physics_similarity=None, 
        performance_metrics=None
    ):
        """
        每个轮次调用，判断是否需要早停
        
        Args:
            current_score (float): 性能指标
            physics_similarity (float, optional): 物理相似度
            performance_metrics (dict, optional): 额外性能指标
        
        Returns:
            bool: 是否触发早停
        """
        # 记录性能历史
        self.performance_history.append(current_score)
        if physics_similarity is not None:
            self.physics_history.append(physics_similarity)
        
        # 首次调用时设置基准分数
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        # 性能趋势分析
        trend_analysis = self._analyze_performance_trend()
        
        # 物理可解释性评估
        physics_check = self._check_physics_interpretability(physics_similarity)
        
        # 判断性能是否改善
        if self.compare_func(current_score, self.best_score) and physics_check:
            # 重置计数器和最佳分数
            self.best_score = current_score
            self.counter = 0
            self.adaptive_patience = self.patience  # 重置动态容忍度
        else:
            # 性能未改善，计数器递增
            self.counter += 1
            
            # 动态调整容忍度
            if trend_analysis == 'decline':
                self.adaptive_patience = max(5, self.adaptive_patience - 2)
            
            print(f"早停监控：{self.counter}/{self.adaptive_patience} 轮无性能改善")
            
            # 判断是否触发早停
            if self.counter >= self.adaptive_patience:
                self.early_stop = True
                print("触发早停：模型性能长期未改善")
                
                # 绘制性能趋势图
                self._visualize_performance_trend()
        
        return self.early_stop
    
    def _analyze_performance_trend(self):
        """
        分析性能趋势
        
        Returns:
            str: 性能趋势('improve', 'stable', 'decline')
        """
        if len(self.performance_history) < self.performance_window:
            return 'stable'
        
        recent_performance = self.performance_history[-self.performance_window:]
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        if trend > self.min_delta:
            return 'improve'
        elif trend < -self.min_delta:
            return 'decline'
        else:
            return 'stable'
    
    def _check_physics_interpretability(self, physics_similarity):
        """
        检查物理可解释性
        
        Args:
            physics_similarity (float): 物理相似度
        
        Returns:
            bool: 是否满足物理可解释性要求
        """
        if physics_similarity is None:
            return True
        
        return physics_similarity >= (1 - self.physics_tolerance)
    
    def _visualize_performance_trend(self):
        """可视化性能趋势"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.title('性能指标趋势')
        plt.plot(self.performance_history, marker='o')
        plt.xlabel('训练轮次')
        plt.ylabel('性能指标')
        
        if self.physics_history:
            plt.subplot(1, 2, 2)
            plt.title('物理相似度趋势')
            plt.plot(self.physics_history, marker='o', color='green')
            plt.xlabel('训练轮次')
            plt.ylabel('物理相似度')
        
        plt.tight_layout()
        plt.savefig('performance_trend.png')
        plt.close()
    
    def reset(self):
        """重置早停状态"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.performance_history = []
        self.physics_history = []
        self.adaptive_patience = self.patience
# 测试代码
if __name__ == "__main__":
    # 模拟训练过程
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("早停类增强测试")
    print("=" * 40)
    
    # 初始化早停
    early_stopper = EarlyStoppingCallback(
        patience=5,      # 容忍轮数
        min_delta=0.01,  # 最小改善阈值
        mode='max'       # 性能模式
    )
    
    # 模拟性能指标序列（包含波动）
    performance_metrics = [
        0.70, 0.72, 0.73, 0.71, 0.74, 
        0.75, 0.76, 0.77, 0.76, 0.75, 
        0.74, 0.73, 0.72, 0.71, 0.70
    ]
    
    # 存储早停过程信息
    best_scores = []
    counters = []
    
    # 测试早停
    for epoch, metric in enumerate(performance_metrics, 1):
        print(f"Epoch {epoch}: 性能指标 = {metric:.2f}")
        
        # 检查是否早停
        stop_flag = early_stopper(metric)
        
        # 记录状态
        best_scores.append(early_stopper.best_score)
        counters.append(early_stopper.counter)
        
        if stop_flag:
            print(f"在第 {epoch} 轮触发早停")
            break
        
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    # 可视化早停过程
    plt.figure(figsize=(15, 5))
    
    # 性能指标曲线
    plt.subplot(1, 3, 1)
    plt.plot(performance_metrics[:len(best_scores)], marker='o')
    plt.title('性能指标')
    plt.xlabel('轮次')
    plt.ylabel('指标值')
    
    # 最佳分数曲线
    plt.subplot(1, 3, 2)
    plt.plot(best_scores, marker='o', color='green')
    plt.title('最佳性能')
    plt.xlabel('轮次')
    plt.ylabel('最佳指标')
    
    # 计数器变化
    plt.subplot(1, 3, 3)
    plt.plot(counters, marker='o', color='red')
    plt.title('早停计数器')
    plt.xlabel('轮次')
    plt.ylabel('连续无改善轮数')
    
    plt.tight_layout()
    plt.savefig('early_stopping_visualization.png')
    plt.show()
    
    # 物理可解释性演示
    def analyze_early_stopping_behavior(best_scores, performance_metrics, counters):
        """分析早停行为"""
        print("\n早停行为分析：")
        print(f"最佳性能指标: {max(best_scores):.4f}")
        print(f"性能指标波动范围: [{min(performance_metrics):.4f}, {max(performance_metrics):.4f}]")
        print(f"最大连续无改善轮数: {max(counters)}")
        
        # 判断早停质量
        performance_std = np.std(performance_metrics)
        stability_index = performance_std / np.mean(performance_metrics)
        
        print(f"性能稳定性指数: {stability_index:.4f}")
        print("稳定性评价：", end=" ")
        if stability_index < 0.05:
            print("高稳定性")
        elif stability_index < 0.1:
            print("中等稳定性")
        else:
            print("低稳定性")
    
    analyze_early_stopping_behavior(best_scores, performance_metrics, counters)
