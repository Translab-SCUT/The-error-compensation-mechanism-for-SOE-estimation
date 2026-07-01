import math
import os
import re
from collections import deque
import numpy as np
import pandas as pd
from calculate_R_Rp_Cp_a_b_c import calculate_paras
from calculate_Emax import cal_Emax


class ASWEKF:
    def __init__(self, n, m, window_size,min_window_size=50, max_window_size=1000, threshold=np.sqrt(5)):
        self.n = n 
        self.m = m  
        self.x = np.zeros(n) 
        self.P = np.eye(n) * 0.1  
        
        
        self.Q = np.diag([1e-12, 1e-4, 1e-1])
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.threshold = threshold
        self.observations = deque(maxlen=self.window_size) 

    def adapt_window_size(self, prediction_error):
       
        if prediction_error < self.threshold:
            new_window_size = max(self.min_window_size, self.window_size - 1)  
        else:
            new_window_size = min(self.max_window_size, self.window_size + 1) 
        if new_window_size != self.window_size:
           
            current_observations = list(self.observations)  
            self.window_size = new_window_size
            self.observations = deque(maxlen=self.window_size) 
          
            for obs in current_observations[-self.window_size:]: 
                self.observations.append(obs)
        return new_window_size
    def predict(self, u, u_p, dt, para, Emax):
      
        R, Rp, Cp, a, b, c = para
        a_adj = np.clip(a, 0.9, 0.999)
       # tao = -1 * dt / Rp / Cp
        
      #  x1_n = self.x[0] - (self.x[2] * u * dt / 3600) / Emax * 100 
      
        x1_n = self.x[0] - (u * dt / 3600) / Emax * 100
     
        tao = -1 * dt / Rp / Cp
        x2_n = self.x[1] * np.exp(tao) + (1 - np.exp(tao)) * Rp * u


      #  x2_n = self.x[1] * (np.exp(tao)) + (1 - np.exp(tao)) * Rp * u 
        x3_n = self.x[2] * a_adj + b * u + c * u_p 
      #  x3_n = self.x[2] + b * u + c * u_p
        self.x = np.array([x1_n, x2_n, x3_n])
       
        F = np.array([[1, 0, 0],
                      [0, np.exp(tao), 0],
                      [0, 0,  a_adj]])

        
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, z, R_cov):
        if len(self.observations) < 2:
            return

       
        # z_eff = np.mean(self.observations)
        z_eff = z

        z_pred = self.h(self.x)
        H = np.array([[0, 0, 1]])

       
        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T / S[0, 0]
      #  print(f"P diag: {np.diag(self.P)}")

        innovation = z_eff - z_pred
       
       # innovation = np.clip(innovation, -2.0, 2.0)
      
        soe_correction = K[0][0] * innovation


      #  print(
         #   f"SOE: {self.x[0]:.4f}, Pred_V: {z_pred:.2f}, Meas_V: {z:.2f}, K_soe: {K[0][0]:.6f}, Innov: {innovation:.4f}")
        self.x = self.x + K.flatten() * innovation

        
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + self.Q  

    def h(self, x):
        return x[2]  

    def add_observation(self, z):
       
        self.observations.append(z) 
     
        return self.observations

    def compute_prediction_error(self, z):
        
        prediction_residual = z - self.h(self.x)
        return np.linalg.norm(prediction_residual)
#############################################################
    def step(self, z, R,u,u_p,dt,paras,Emax):
        
        self.predict(u,u_p,dt,paras,Emax)
     
        #self.observations.append(z)
        #self.add_observation(z) 
      
        prediction_error = self.compute_prediction_error(z)
        self.adapt_window_size(prediction_error)
        self.add_observation(z)
      
        self.update(z,R)
        self.x[0] = np.clip(self.x[0], 0, 100)
        return self.x


def extract_number_for_sort(filename):
   
    match = re.search(r'\d+', filename)
    if match:
      
        return (filename[:match.start()], int(match.group()), filename[match.end():])
    else:
     
        return (filename, None)


def ekf_for_soe(folder_path):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for file_name in sorted(file_list, key=extract_number_for_sort):
        file_path = os.path.join(folder_path, file_name)
        print(f"正在处理: {file_name}")

   
        try:
            df = pd.read_csv(file_path, encoding='gbk')

          
            if len(df) < 110:
              
                continue

           

          
            time_col = '数据时间'
            df[time_col] = pd.to_datetime(df[time_col])
            df['delta_t'] = df[time_col].diff().dt.total_seconds()

          
            df = df[df['delta_t'] != 0].reset_index(drop=True)

          
            df['delta_t'] = df[time_col].diff().dt.total_seconds()

        
            default_dt = 10.0 
            df['delta_t'] = df['delta_t'].fillna(default_dt)

           
            df = df[df['delta_t'] < 30].reset_index(drop=True)
            df['温度极差'] = df['最高温度值'] - df['最低温度值']
        
            window = 5 
            df['总电流'] = df['总电流'].rolling(window=window, min_periods=1, center=True).mean()
            df['总电压'] = df['总电压'].rolling(window=window, min_periods=1, center=True).mean()

         
            R, RP, CP, list_a, list_b, list_c = calculate_paras(df)




        

            n, m, R_noise, window_size = 3, 1, 0.01, 50
            aswekf = ASWEKF(n, m, window_size)
           
            initial_soc = float(df['SOE_Proposed'].iloc[0])
         
            actual_v = float(df['总电压'].iloc[0])
            aswekf.x = np.array([initial_soc, 0.0, actual_v - initial_soc])

            soe_results = []
            R_noise = 0.01

        
            for k in range(0, len(df) - 1):
                u_prev = df['总电流'].iloc[k]
                u_curr = df['总电流'].iloc[k + 1]
                z_curr = df['总电压'].iloc[k + 1] 
                t_max = df['最高温度值'].iloc[k + 1]
                t_delt = df['温度极差'].iloc[k + 1]  
                dt = df['delta_t'].iloc[k + 1]

             
                if "505Ah" in file_path:
                    c_cap, t_cells, u_n = 505.0, 324, 3.2  
                else:
                    c_cap, t_cells, u_n = 160.0, 91, 3.6 
              

               
                E_max = cal_Emax(u_curr, t_max, t_delt, c_cap, t_cells, u_n,aswekf.x[0])
                current_paras = [R, RP, CP, list_a[k], list_b[k], list_c[k]]
                theoretical_drop = (u_curr * dt / 3600) / E_max * 100
               

                
                old_soe = aswekf.x[0]
                if abs(u_curr) < 1.0:
                    
                    R_noise_dynamic = 0.01  
                else:
                   
                    R_noise_dynamic = 2


                state = aswekf.step(z_curr, R_noise_dynamic, u_curr, u_prev, dt, current_paras, E_max)
               
                actual_change = state[0] - old_soe



                soe_results.append(state[0])

       
            df['ASWEKF_SOE'] = [initial_soc] + soe_results
            df.to_csv(file_path, index=False, encoding='gbk')
            

        except Exception as e:
           
            continue













#plt.show()
#plt.plot(ut)
#plt.show()
if __name__ == "__main__":
    path = r'E:\SOE\实车数据集处理\切分片段结果\电车公司1_505Ah\SOE_Comparison_Output'
    ekf_for_soe(path)
