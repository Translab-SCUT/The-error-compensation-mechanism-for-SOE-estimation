import pandas as pd
import numpy as np
import os
from glob import glob


def process_missing_values(base_dir):
    """
    base_dir: 存放排序去重后CSV的根目录
    """
    # 查找所有子文件夹下的 CSV 文件
    file_list = glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

    # 根据你广州公交数据集定义的列名（请确保合并时的列名一致）
    # 平稳变化的数据（论文要求：前向填充）
    steady_cols = [
        '总电压', 'SOC', '最高温度值', '最低温度值',
        '电池单体电压最高值', '电池单体电压最低值'
    ]

    # 剧烈变化的数据（论文要求：样条插值）
    volatile_cols = ['车速', '总电流']

    for file_path in file_list:
        print(f"正在处理缺失值与异常值: {file_path}")

        try:
            # 1. 读取数据 (low_memory=False 防止列名推断错误)
            df = pd.read_csv(file_path, low_memory=False)

            # 2. 异常值处理：将不合理的 0 值转为 NaN（缺失值）
            # 电压和SOC在运行中不可能为0，这些通常是采样错误
            cols_to_fix_zero = ['总电压', 'SOC', '电池单体电压最高值', '电池单体电压最低值']
            for col in cols_to_fix_zero:
                if col in df.columns:
                    # 将0替换为NaN，方便后续填充
                    df[col] = df[col].replace(0, np.nan)

            # 3. 填充处理 - 逻辑 A: 平稳数据使用前向填充 (ffill)
            for col in steady_cols:
                if col in df.columns:
                    df[col] = df[col].ffill()

            # 4. 填充处理 - 逻辑 B: 剧烈数据使用样条插值 (Spline Interpolation)
            # 论文提到车速和放电电流波动大，插值比前向填充更符合物理真实性
            for col in volatile_cols:
                if col in df.columns:
                    # limit_direction='both' 确保开头缺失也能补全
                    # order=2 或 3 为样条插值，若数据量极小可用 linear
                    try:
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    except:
                        df[col] = df[col].ffill().bfill()

            # 5. 最后兜底：如果还有开头缺失的 NaN，统一用后向填充补齐
            df = df.ffill().bfill()

            # 6. 保存处理后的结果
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"  √ 缺失值填充完成")

        except Exception as e:
            print(f"  × 处理失败 {file_path}: {e}")


# --- 路径配置 ---
merged_data_dir = r"E:\SOE\实车数据集处理\广州公交数据合并"
process_missing_values(merged_data_dir)