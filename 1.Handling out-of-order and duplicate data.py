import pandas as pd
import os
from glob import glob


def sort_and_deduplicate(base_dir):
    """
    
    """
   
    file_list = glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

    for file_path in file_list:
       

        try:
            
          #  df = pd.read_csv(file_path)
            
            df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)

          
          
            df['数据时间'] = pd.to_datetime(df['数据时间'])

          
            df = df.sort_values(by='数据时间', ascending=True)

            
            initial_rows = len(df)
            df = df.drop_duplicates()
            final_rows = len(df)

            if initial_rows > final_rows:
                print(f"  Number of duplicate rows removed: {initial_rows - final_rows}")

           
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"  current_rows: {final_rows}")

        except Exception as e:
            print(f"  Failed to process file {file_path}: {e}")



merged_data_dir = r"E:\SOE\data_processing\data merging"
sort_and_deduplicate(merged_data_dir)
