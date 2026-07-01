import pandas as pd
import os
from glob import glob


def split_segments_with_identity(base_dir, output_root):
   
    os.makedirs(output_root, exist_ok=True)

    file_list = glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

    for file_path in file_list:
        file_name = os.path.basename(file_path).replace('.csv', '')

        
        if "bus" in file_path:
            company_label = "bus_505Ah"
        elif "taxi" in file_path:
            company_label = "taxi_160Ah"
        else:
            company_label = "unknown"

        charge_dir = os.path.join(output_root, company_label, "charging segment")
        discharge_dir = os.path.join(output_root, company_label, "discharging segment")
        os.makedirs(charge_dir, exist_ok=True)
        os.makedirs(discharge_dir, exist_ok=True)

        

        try:
            df = pd.read_csv(file_path)

           
            if '驱动电机转矩' in df.columns:
                df.loc[(df['充电状态'] == 1) & ((df['车速'] > 0) | (df['驱动电机转矩'] > 5)), '充电状态'] = 3
            else:
                df.loc[(df['充电状态'] == 1) & (df['车速'] > 0), '充电状态'] = 3

            
            df.loc[(df['充电状态'] == 3) & (df['总电流'] < -5) & (df['车速'] == 0), '充电状态'] = 1

            
            charge_df = df[df['充电状态'] == 1].copy()
            if not charge_df.empty:
               
                charge_df['time_gap'] = pd.to_datetime(charge_df['数据时间']).diff().dt.total_seconds() > 300
                charge_df['soc_gap'] = charge_df['SOC'].diff() < -1
                charge_df['group'] = (charge_df['time_gap'] | charge_df['soc_gap']).cumsum()

                for g_id, segment in charge_df.groupby('group'):
                    delta_soc = segment['SOC'].max() - segment['SOC'].min()
                    
                    if len(segment) > 50 and delta_soc > 5:
                        save_name = f"{file_name}_Charge_{g_id}.csv"
                        segment.to_csv(os.path.join(charge_dir, save_name), index=False, encoding='utf-8-sig')


            
            discharge_df = df[df['充电状态'] == 3].copy()
            if not discharge_df.empty:
             
                discharge_df['dt_obj'] = pd.to_datetime(discharge_df['数据时间'])

                
                discharge_df['time_gap'] = discharge_df['dt_obj'].diff().dt.total_seconds() > 300

                
                discharge_df['soc_jump'] = discharge_df['SOC'].diff().abs() > 5

          
                discharge_df['soc_back'] = discharge_df['SOC'].diff() > 2

                
                discharge_df['group'] = (
                            discharge_df['time_gap'] | discharge_df['soc_jump'] | discharge_df['soc_back']).cumsum()

                for g_id, segment in discharge_df.groupby('group'):
                  
                    if len(segment) > 500:
                        delta_soc = segment['SOC'].max() - segment['SOC'].min()
                        if delta_soc > 2: 
                            save_name = f"{file_name}_Discharge_{g_id}.csv"
                            segment.to_csv(os.path.join(discharge_dir, save_name), index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    


# --- 路径配置 ---
input_dir = r"E:\SOE\data_processing\taxi_data"
output_path = r"E:\SOE\data_processing\Taxi segment extraction results"

split_segments_with_identity(input_dir, output_path)
