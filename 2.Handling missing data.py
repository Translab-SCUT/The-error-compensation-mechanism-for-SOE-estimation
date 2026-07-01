import pandas as pd
import numpy as np
import os
from glob import glob


def process_missing_values(base_dir):
  
    file_list = glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

    
    steady_cols = [
        '总电压', 'SOC', '最高温度值', '最低温度值',
        '电池单体电压最高值', '电池单体电压最低值'
    ]

   
    volatile_cols = ['车速', '总电流']

    for file_path in file_list:
        

        try:
            
            df = pd.read_csv(file_path, low_memory=False)

           
            cols_to_fix_zero = ['总电压', 'SOC', '电池单体电压最高值', '电池单体电压最低值']
            for col in cols_to_fix_zero:
                if col in df.columns:
                   
                    df[col] = df[col].replace(0, np.nan)

           
            for col in steady_cols:
                if col in df.columns:
                    df[col] = df[col].ffill()

           
            for col in volatile_cols:
                if col in df.columns:
                    
                    try:
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    except:
                        df[col] = df[col].ffill().bfill()

           
            df = df.ffill().bfill()

           
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"  √ 缺失值填充完成")

        except Exception as e:
            print(f"  × 处理失败 {file_path}: {e}")



merged_data_dir = r"E:\SOE\实车数据集处理\广州公交数据合并"
process_missing_values(merged_data_dir)
