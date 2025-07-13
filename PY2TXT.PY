import os
import shutil

# 源目录和目标目录（使用原始字符串避免转义问题，末尾不加斜杠）
source_folder = r'E:\20250711电机小论文'  # 原始字符串标记 r'' 避免 \ 转义
target_folder = r'E:\20250711电机小论文\PY2TXT'

# 创建目标文件夹（如果不存在）
os.makedirs(target_folder, exist_ok=True)

# 统计处理的文件数量
count = 0

# 递归遍历源目录下的所有文件（包括子目录）
for root, dirs, files in os.walk(source_folder):
    for filename in files:
        if filename.endswith('.py'):
            # 构建源文件完整路径
            py_file_path = os.path.join(root, filename)
            # 构建目标文件相对路径（保持子目录结构）
            relative_path = os.path.relpath(root, source_folder)
            target_dir = os.path.join(target_folder, relative_path)
            # 创建目标子目录（如果不存在）
            os.makedirs(target_dir, exist_ok=True)
            # 构建目标.txt文件路径
            txt_filename = filename.replace('.py', '.txt')
            txt_file_path = os.path.join(target_dir, txt_filename)
            
            # 复制文件内容（尝试多种编码容错）
            try:
                # 尝试UTF-8编码读取
                with open(py_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    # 尝试GBK编码（中文Windows常见编码）
                    with open(py_file_path, 'r', encoding='gbk') as f:
                        content = f.read()
                except:
                    # 最终尝试忽略错误编码
                    with open(py_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
            
            # 写入.txt文件
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            count += 1
            print(f"已处理：{py_file_path} -> {txt_file_path}")

if count == 0:
    print(f"未找到任何.py文件，请检查源目录：{source_folder}")
else:
    print(f"处理完成！共转换 {count} 个.py文件到目标目录：{target_folder}")