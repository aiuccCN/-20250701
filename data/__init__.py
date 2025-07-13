# data/__init__.py

# 导入数据加载器和预处理器
from .university_of_ottawa_loader import UniversityOfOttawaDataLoader
from configs.config import config, Config
# 定义包的公开接口
__all__ = [
    'UniversityOfOttawaDataLoader', 
]

# 可选：添加包级别的说明文档
"""
University of Ottawa电机故障数据集加载和预处理模块

包含:
- UniversityOfOttawaDataLoader: 数据集加载器

"""
