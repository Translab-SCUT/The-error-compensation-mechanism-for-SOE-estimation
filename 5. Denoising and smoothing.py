import pandas as pd
import numpy as np
import os
from glob import glob
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib

# 设置绘图字体（防止中文乱码）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def process_and_visualize_denoising(root_dir, target_verify_file):
    """
    root_dir: 片段根目录
    target_verify_file: 想要用来对比查看的特定文件完整路径
    """
    file_list = glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    if not file_list:
        print("未找到CSV文件，请检查路径。")
        return

    sg_cols = ['总电流', '总电压', '电池单体电压最高值', '电池单体电压最低值', '车速']
    moving_avg_cols = ['最高温度值', '最低温度值']

    print(f"开始全量平滑处理，共 {len(file_list)} 个文件...")

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            if len(df) < 15: continue

            # 如果是目标验证文件，保留一份原始备份用于绘图
            is_target = (os.path.abspath(file_path) == os.path.abspath(target_verify_file))
            if is_target:
                df_raw = df.copy()
                print(f"--- 捕捉到验证文件，准备绘图对比: {os.path.basename(file_path)} ---")

            # --- 执行平滑逻辑 ---
            # 1. SG 滤波 (电参数)
            for col in sg_cols:
                if col in df.columns:
                    df[col] = df[col].interpolate().ffill().bfill()
                    df[col] = savgol_filter(df[col], window_length=11, polyorder=2)

            # 2. 滑动平均 (温度)
            for col in moving_avg_cols:
                if col in df.columns:
                    df[col] = df[col].interpolate().rolling(window=5, center=True).mean().ffill().bfill()

            # --- 如果是目标文件，弹出对比图 ---
            if is_target:
                fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

                # 电流对比
                axes[0].plot(df_raw['总电流'], color='silver', label='原始电流 (带噪)', alpha=0.7)
                axes[0].plot(df['总电流'], color='red', label='平滑后 (SG滤波)', linewidth=1.2)
                axes[0].set_title('总电流去噪对比 (Target Verification)')
                axes[0].legend()

                # 电压对比
                axes[1].plot(df_raw['总电压'], color='lightblue', label='原始电压', alpha=0.7)
                axes[1].plot(df['总电压'], color='blue', label='平滑后 (SG滤波)', linewidth=1.2)
                axes[1].set_title('总电压去噪对比')
                axes[1].legend()

                # 温度对比
                axes[2].plot(df_raw['最高温度值'], color='peachpuff', label='原始温度 (阶梯)', alpha=0.8,
                             drawstyle='steps-post')
                axes[2].plot(df['最高温度值'], color='darkorange', label='平滑后 (滑动平均)', linewidth=2)
                axes[2].set_title('最高温度平滑对比')
                axes[2].legend()

                plt.tight_layout()
                plt.show()

            # 保存结果
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"处理 {os.path.basename(file_path)} 失败: {e}")

    print("\n--- 所有片段去噪平滑完成并已覆盖保存 ---")


# --- 路径配置 ---
output_root = r"E:\SOE\实车数据集处理\切分片段结果"
# 你指定的特定 CSV 文件路径
target_file = r"E:\SOE\实车数据集处理\切分片段结果\出租车公司2_160Ah\放电片段\20240530153340_LGB61YEA0KS076518_20230401000000_20230601000000_Discharge_10.csv"

if __name__ == "__main__":
    process_and_visualize_denoising(output_root, target_file)