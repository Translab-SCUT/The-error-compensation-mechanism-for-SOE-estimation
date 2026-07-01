import pandas as pd
import os
from glob import glob


def sort_and_deduplicate(base_dir):
    """
    base_dir: 存放合并后CSV的根目录 (例如: F:\广州公交处理结果\1_已合并Sheet数据)
    """
    # 查找所有子文件夹下的 CSV 文件
    file_list = glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

    for file_path in file_list:
        print(f"正在进行乱序重复处理: {file_path}")

        try:
            # 1. 读取数据
          #  df = pd.read_csv(file_path)
            # 修改这一行
            df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)

            # 2. 转换时间格式 (确保能按时间先后排序)
            # 假设你的时间列名是 '数据时间'
            df['数据时间'] = pd.to_datetime(df['数据时间'])

            # 3. 排序：按时间先后顺序排列
            df = df.sort_values(by='数据时间', ascending=True)

            # 4. 去重：删除完全重复的行
            initial_rows = len(df)
            df = df.drop_duplicates()
            final_rows = len(df)

            if initial_rows > final_rows:
                print(f"  已删除重复行数: {initial_rows - final_rows}")

            # 5. 重置索引并保存
            # 注意：覆盖原始合并后的CSV，或者另存为一个新目录
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"  排序去重完成，当前行数: {final_rows}")

        except Exception as e:
            print(f"  处理文件 {file_path} 失败: {e}")


# --- 路径配置 ---
merged_data_dir = r"E:\SOE\实车数据集处理\广州公交数据合并"
sort_and_deduplicate(merged_data_dir)