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
        self.n = n  # 状态维度
        self.m = m  # 观测维度
        self.x = np.zeros(n)  # 初始状态向量
        self.P = np.eye(n) * 0.1  # 初始协方差矩阵
        # --- 新增：过程噪声矩阵 Q ---
        # Q[0,0]: SOE 的噪声，控制 SOE 修正的灵活性
        # Q[1,1], Q[2,2]: 极化电压噪声
        self.Q = np.diag([1e-12, 1e-4, 1e-1])
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.threshold = threshold
        self.observations = deque(maxlen=self.window_size)  # 观测数据缓冲区

    def adapt_window_size(self, prediction_error):
        # 动态调整窗口大小
        if prediction_error < self.threshold:
            new_window_size = max(self.min_window_size, self.window_size - 1)  # 减小窗口
        else:
            new_window_size = min(self.max_window_size, self.window_size + 1)  # 增大窗口
        if new_window_size != self.window_size:
            #print(f"窗口大小改变: {self.window_size} -> {new_window_size}")
            current_observations = list(self.observations)  # 保存当前观测值
            self.window_size = new_window_size
            self.observations = deque(maxlen=self.window_size)  # 重新创建 deque
            # 将之前的观测值重新添加到新的 deque 中
            for obs in current_observations[-self.window_size:]:  # 只保留最近的元素
                self.observations.append(obs)
        return new_window_size
    def predict(self, u, u_p, dt, para, Emax):
        # 参数设置
        R, Rp, Cp, a, b, c = para
        a_adj = np.clip(a, 0.9, 0.999)
       # tao = -1 * dt / Rp / Cp
        # 状态转移方程
      #  x1_n = self.x[0] - (self.x[2] * u * dt / 3600) / Emax * 100 # 状态1
        # 如果 u > 0 是放电，SOE 下降
        x1_n = self.x[0] - (u * dt / 3600) / Emax * 100
        # 2. 极化电压
        tao = -1 * dt / Rp / Cp
        x2_n = self.x[1] * np.exp(tao) + (1 - np.exp(tao)) * Rp * u


      #  x2_n = self.x[1] * (np.exp(tao)) + (1 - np.exp(tao)) * Rp * u  # 状态2
        x3_n = self.x[2] * a_adj + b * u + c * u_p  # 状态3
      #  x3_n = self.x[2] + b * u + c * u_p
        self.x = np.array([x1_n, x2_n, x3_n])
        # 线性化状态转移函数
        F = np.array([[1, 0, 0],
                      [0, np.exp(tao), 0],
                      [0, 0,  a_adj]])

        # 预测协方差
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, z, R_cov):
        if len(self.observations) < 2:
            return

        # 建议直接使用当前观测 z，或者对窗口数据进行加权
        # z_eff = np.mean(self.observations)
        z_eff = z

        z_pred = self.h(self.x)
        H = np.array([[0, 0, 1]])

        # 标准 EKF 更新步骤
        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T / S[0, 0]
      #  print(f"P diag: {np.diag(self.P)}")

        innovation = z_eff - z_pred
        # 限制在 +/- 1V 以内
       # innovation = np.clip(innovation, -2.0, 2.0)
        # 1. 计算本次电压校准对 SOE 造成的改变量
        soe_correction = K[0][0] * innovation


      #  print(
         #   f"SOE: {self.x[0]:.4f}, Pred_V: {z_pred:.2f}, Meas_V: {z:.2f}, K_soe: {K[0][0]:.6f}, Innov: {innovation:.4f}")
        self.x = self.x + K.flatten() * innovation

        # 更新协方差 (使用约瑟夫形式更稳定)
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + self.Q  # 适当的过程噪声

    def h(self, x):
        return x[2]  # 观测函数，返回 x3

    def add_observation(self, z):
        #print(f"加入的观测值: {z}")  # 打印观测值
        self.observations.append(z)  # 添加新观测
        #print(f"观测长度 : {len(self.observations)}")
        return self.observations

    def compute_prediction_error(self, z):
        # 计算预测误差
        prediction_residual = z - self.h(self.x)
        return np.linalg.norm(prediction_residual)
#############################################################
    def step(self, z, R,u,u_p,dt,paras,Emax):
        # 进行一步EKF估计predict(u_current, u_previous, dt, paras, E_max)
        self.predict(u,u_p,dt,paras,Emax)
        # 添加新观测到缓冲区
        #self.observations.append(z)
        #self.add_observation(z)  # 添加新观测
        # 计算预测误差
        prediction_error = self.compute_prediction_error(z)
        self.adapt_window_size(prediction_error)
        self.add_observation(z)  # 添加新观测
        # 状态更新
        self.update(z,R)
        self.x[0] = np.clip(self.x[0], 0, 100)
        return self.x

# 提取文件名中的数字并返回用于排序的元组
def extract_number_for_sort(filename):
    # 使用正则表达式找到文件名中的数字，并返回第一个找到的数字（假设文件名中只有一个数字用于排序）
    match = re.search(r'\d+', filename)
    if match:
        # 返回数字前面部分的字符串和数字本身组成的元组，用于排序
        return (filename[:match.start()], int(match.group()), filename[match.end():])
    else:
        # 如果没有找到数字，返回一个特殊的值（例如None）
        return (filename, None)


def ekf_for_soe(folder_path):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for file_name in sorted(file_list, key=extract_number_for_sort):
        file_path = os.path.join(folder_path, file_name)
        print(f"正在处理: {file_name}")

        # 1. 预读数据，检查长度
        try:
            df = pd.read_csv(file_path, encoding='gbk')

            # 如果数据行数太少（比如不足110行，考虑到RLS需要前100个点收敛）
            if len(df) < 110:
                print(f"跳过文件: {file_name} (数据量过少: {len(df)}行)")
                continue

            print(f"正在处理: {file_name}")

            # 1. 准备时间间隔 dt
            # 1. 先计算 delta_t
            time_col = '数据时间'
            df[time_col] = pd.to_datetime(df[time_col])
            df['delta_t'] = df[time_col].diff().dt.total_seconds()

            # 2. 【核心修改】剔除重复时间行（即 delta_t 为 0 的行）
            # 我们只保留 delta_t > 0 的行，这样就去掉了时间相同但数据部分不同的冗余行
            df = df[df['delta_t'] != 0].reset_index(drop=True)

            # 3. 重新计算剔除后的 delta_t（因为删行后间隔会改变）
            df['delta_t'] = df[time_col].diff().dt.total_seconds()

            # 4. 填充第一行和处理断传
            default_dt = 10.0  # 既然你已知是 10s 采样
            df['delta_t'] = df['delta_t'].fillna(default_dt)

            # 5. 【关键修改】放宽阈值，确保正常的 10s 不被过滤
            # 建议设为 30s，这样 10s, 20s 的正常波动都能进来，只有超过 30s 的断层才剔除
            df = df[df['delta_t'] < 30].reset_index(drop=True)
            df['温度极差'] = df['最高温度值'] - df['最低温度值']
            # --- 新增：数据平滑处理 ---
            window = 5  # 平滑窗口大小，建议 5-10
            df['总电流'] = df['总电流'].rolling(window=window, min_periods=1, center=True).mean()
            df['总电压'] = df['总电压'].rolling(window=window, min_periods=1, center=True).mean()

            # 3. 运行 RLS 获取参数 (增加异常捕获，防止 calculate_paras 内部崩溃)
            R, RP, CP, list_a, list_b, list_c = calculate_paras(df)




            # 3. 初始化 EKF

            n, m, R_noise, window_size = 3, 1, 0.01, 50
            aswekf = ASWEKF(n, m, window_size)
            # 建议：获取数据集第一行的 SOC 作为初始 SOE
            initial_soc = float(df['SOE_Proposed'].iloc[0])
            # 初始值设为第一行实际电压
            actual_v = float(df['总电压'].iloc[0])
            aswekf.x = np.array([initial_soc, 0.0, actual_v - initial_soc])

            soe_results = []
            R_noise = 0.01

            # 4. 迭代
            for k in range(0, len(df) - 1):
                u_prev = df['总电流'].iloc[k]
                u_curr = df['总电流'].iloc[k + 1]
                z_curr = df['总电压'].iloc[k + 1]  # 这里的电压作为校准基准
                t_max = df['最高温度值'].iloc[k + 1]
                t_delt = df['温度极差'].iloc[k + 1]  # 使用刚刚计算出的极差
                dt = df['delta_t'].iloc[k + 1]

                # 假设通过路径判断车型
                if "505Ah" in file_path:
                    c_cap, t_cells, u_n = 505.0, 324, 3.2  # 电车参数
                else:
                    c_cap, t_cells, u_n = 160.0, 91, 3.6  # 出租车参数
                print(f"--- 调试：当前文件识别容量 c_cap = {c_cap} ---")

                # 调用修改后的函数
                E_max = cal_Emax(u_curr, t_max, t_delt, c_cap, t_cells, u_n,aswekf.x[0])
                current_paras = [R, RP, CP, list_a[k], list_b[k], list_c[k]]
                theoretical_drop = (u_curr * dt / 3600) / E_max * 100
                print(f"Step {k} | 当前 E_max: {E_max:.2f}, 当前电流: {u_curr:.2f}, dt: {dt:.1f}")

                # 记录运行前的 SOE
                old_soe = aswekf.x[0]
                if abs(u_curr) < 1.0:
                    # 电流极小时（比如停车或信号红灯），电池处于准平衡态，电压非常准
                    R_noise_dynamic = 0.01  # 甚至可以更小，强制拉回漂移
                else:
                    # 正常行驶时，由于极化和噪声，给观测留一点余地
                    R_noise_dynamic = 2


                state = aswekf.step(z_curr, R_noise_dynamic, u_curr, u_prev, dt, current_paras, E_max)
                # --- 新增：计算实际变化量并打印 ---
                actual_change = state[0] - old_soe



                soe_results.append(state[0])

            # 5. 回写
            df['ASWEKF_SOE'] = [initial_soc] + soe_results
            df.to_csv(file_path, index=False, encoding='gbk')
            print(f"-> {file_name} 已完成 SOE 估计并保存至最后一列。")

        except Exception as e:
            print(f"跳过文件: {file_name} (运行异常: {e})")
            continue













#plt.show()
#plt.plot(ut)
#plt.show()
if __name__ == "__main__":
    path = r'E:\SOE\实车数据集处理\切分片段结果\电车公司1_505Ah\SOE_Comparison_Output'
    ekf_for_soe(path)