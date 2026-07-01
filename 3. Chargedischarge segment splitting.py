import pandas as pd
import os
from glob import glob


def split_segments_with_identity(base_dir, output_root):
    # 建立输出目录（按公司分层，方便后续SOH计算）
    # 目录结构：输出根目录 -> 公司名 -> 充电/放电片段
    os.makedirs(output_root, exist_ok=True)

    file_list = glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

    for file_path in file_list:
        file_name = os.path.basename(file_path).replace('.csv', '')

        # --- 0. 识别身份 (根据路径名判断) ---
        # 假设你的原始文件夹名字包含 "电车有限公司1" 或 "出租汽车有限公司2"
        if "电车有限公司1" in file_path:
            company_label = "电车公司1_505Ah"
        elif "出租汽车有限公司2" in file_path:
            company_label = "出租车公司2_160Ah"
        else:
            company_label = "未知公司"

        # 创建对应的存放路径
        charge_dir = os.path.join(output_root, company_label, "充电片段")
        discharge_dir = os.path.join(output_root, company_label, "放电片段")
        os.makedirs(charge_dir, exist_ok=True)
        os.makedirs(discharge_dir, exist_ok=True)

        print(f"正在处理: {file_name} (归类至: {company_label})")

        try:
            df = pd.read_csv(file_path)

            # --- 1. 状态修正逻辑 ---
            # 类型 I: 状态显示充电(1)但有车速/转矩 -> 修正为放电(3)
            if '驱动电机转矩' in df.columns:
                df.loc[(df['充电状态'] == 1) & ((df['车速'] > 0) | (df['驱动电机转矩'] > 5)), '充电状态'] = 3
            else:
                df.loc[(df['充电状态'] == 1) & (df['车速'] > 0), '充电状态'] = 3

            # 类型 II: 状态显示放电(3)但电流为负且车速为0 -> 修正为充电(1)
            df.loc[(df['充电状态'] == 3) & (df['总电流'] < -5) & (df['车速'] == 0), '充电状态'] = 1

            # --- 2. 充电片段切分 ---
            charge_df = df[df['充电状态'] == 1].copy()
            if not charge_df.empty:
                # SOC 连续性判断：如果时间差 > 300秒 或 SOC 突然跳变，认为不连续
                charge_df['time_gap'] = pd.to_datetime(charge_df['数据时间']).diff().dt.total_seconds() > 300
                charge_df['soc_gap'] = charge_df['SOC'].diff() < -1
                charge_df['group'] = (charge_df['time_gap'] | charge_df['soc_gap']).cumsum()

                for g_id, segment in charge_df.groupby('group'):
                    delta_soc = segment['SOC'].max() - segment['SOC'].min()
                    # 过滤条件：行数 > 50 且 SOC 变化 > 5%
                    if len(segment) > 50 and delta_soc > 5:
                        save_name = f"{file_name}_Charge_{g_id}.csv"
                        segment.to_csv(os.path.join(charge_dir, save_name), index=False, encoding='utf-8-sig')


            # --- 3. 修正后的放电片段切分 ---
            discharge_df = df[df['充电状态'] == 3].copy()
            if not discharge_df.empty:
                # 转换为时间格式
                discharge_df['dt_obj'] = pd.to_datetime(discharge_df['数据时间'])

                # 条件1：时间裂缝 > 300秒 (断传判断)
                discharge_df['time_gap'] = discharge_df['dt_obj'].diff().dt.total_seconds() > 300

                # 条件2：SOC 异常跳变 (例如下降超过 5% 但时间很短，可能是数据丢失)
                discharge_df['soc_jump'] = discharge_df['SOC'].diff().abs() > 5

                # 条件3：SOC 异常回升 (你的原逻辑)
                discharge_df['soc_back'] = discharge_df['SOC'].diff() > 2

                # 综合判断：只要满足任何一个，就切分成新片段
                discharge_df['group'] = (
                            discharge_df['time_gap'] | discharge_df['soc_jump'] | discharge_df['soc_back']).cumsum()

                for g_id, segment in discharge_df.groupby('group'):
                    # 过滤条件：行数 > 500 且 确实存在能量消耗
                    if len(segment) > 500:
                        delta_soc = segment['SOC'].max() - segment['SOC'].min()
                        if delta_soc > 2:  # 至少要有一定的电量下降才有辨识意义
                            save_name = f"{file_name}_Discharge_{g_id}.csv"
                            segment.to_csv(os.path.join(discharge_dir, save_name), index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"文件 {file_name} 处理出错: {e}")

    print("\n--- 所有片段切分并分类完成 ---")


# --- 路径配置 ---
input_dir = r"E:\SOE\实车数据集处理\广州公交数据合并\广州公交集团广交出租汽车有限公司2"
output_path = r"E:\SOE\实车数据集处理\出租车切分片段结果"

split_segments_with_identity(input_dir, output_path)