import math
import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter

matplotlib.use('TkAgg')
np.random.seed(0)
torch.manual_seed(0)
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

from PatchFormer import PatchFormer
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['text.usetex'] = False


# ==========================================
# 1. 工具类与辅助函数
# ==========================================
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0] = 1e-5

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform_label(self, data):
        # 专门针对标签列（最后一列 SOE_Error）的反归一化
        return (data * self.std[-1]) + self.mean[-1]


def plot_loss_data(train_loss, val_loss):
    """绘制训练和验证的 Loss 曲线"""
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss', marker='o')
    plt.plot(val_loss, label='Val Loss', marker='s')
    plt.title("Loss Results Plot")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def calculate_metrics(y_true, y_pred):
    """计算多维度评价指标"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    # MAPE 增加保护，防止除以 0
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-5))) * 100
    me = np.max(np.abs(y_true - y_pred))
    return mae, rmse, mape, me


# ==========================================
# 2. 数据集类：Lazy Loading 解决内存爆炸
# ==========================================
class LazyTimeSeriesDataset(Dataset):
    def __init__(self, file_list, config, scaler, feature_cols, label_col):
        self.config = config
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.all_data = []  # 存储每个文件的 numpy 矩阵（原始大小）
        self.indices = []  # 存储 (文件索引, 行索引)

        for f_idx, f in enumerate(tqdm(file_list, desc="加载并预处理文件")):
            df = pd.read_csv(f, encoding='gbk')
          #  df['I_diff'] = df['Enhanced_Current'].diff().fillna(0)
          #  df[label_col] = df[label_col].rolling(window=5, min_periods=1).mean()
            df[feature_cols] = df[feature_cols].rolling(window=3, min_periods=1).mean()


            valid_cols = feature_cols + label_col
            data = df[valid_cols].values.astype(np.float32)

            # 关键：在这里进行归一化
            data_scaled = self.scaler.transform(data)
            self.all_data.append(data_scaled)

            # 计算该文件能切分多少个序列
            num_samples = len(data_scaled) - config.seq_len - config.pre_len + 1
            for i in range(num_samples):
                self.indices.append((f_idx, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        f_idx, start_idx = self.indices[index]
        data = self.all_data[f_idx]

        # 切片提取 (X: 特征, Y: 标签)
        seq = data[start_idx: start_idx + self.config.seq_len, :len(self.feature_cols)]
        label = data[start_idx + self.config.seq_len: start_idx + self.config.seq_len + self.config.pre_len, -1:]

        return torch.FloatTensor(seq), torch.FloatTensor(label)


# ==========================================
# 3. 主程序逻辑
# ==========================================
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 文件切分 (8:1:1)
    all_files = sorted(glob.glob(os.path.join(config.data_path, "*.csv")))
    if not all_files: return print("未找到CSV文件")
    import random

    random.seed(42)
    random.shuffle(all_files)

    n_files = len(all_files)
    train_end = int(n_files * 0.8)
    val_end = int(n_files * 0.9)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # 【新增】保存测试集名单，方便后续对齐绘制轨迹图
    with open("test_files_list.txt", "w") as f:
        for item in test_files:
            f.write("%s\n" % item)

    print(f"对齐后的测试集文件数: {len(test_files)}")

    feature_cols = ['电池单体电压最高值','电池单体电压最低值','最高温度值','温度极差',
                    'SOC', '总电压','总电流','SOH',  'Enhanced_Current','Enhanced_Speed']
    label_col = ['SOE_Error']

   # label_col = ['SOE_Proposed']

    # 2. 计算归一化参数 (只用训练集)
    print("正在分析训练集分布...")
    sample_data = []
    for f in tqdm(train_files, desc="归一化分析"):
        df_temp = pd.read_csv(f, encoding='gbk')

        # 【关键】必须执行和 Dataset 里面一模一样的预处理逻辑
       # df_temp['I_diff'] = df_temp['Enhanced_Current'].diff().fillna(0)
       # df_temp[label_col] = df_temp[label_col].rolling(window=5, min_periods=1).mean()
        df_temp[feature_cols] = df_temp[feature_cols].rolling(window=3, min_periods=1).mean()


        # 填补因 rolling 产生的少量 NaN（如果有的话）
        data_values = df_temp[feature_cols + label_col].fillna(method='bfill').fillna(method='ffill').values
        sample_data.append(data_values)

    scaler = StandardScaler()
    scaler.fit(np.vstack(sample_data))
    print("归一化参数计算完成。")
    del sample_data

    # 3. 构建 Dataset 和 Loader
    train_set = LazyTimeSeriesDataset(train_files, config, scaler, feature_cols, label_col)
    val_set = LazyTimeSeriesDataset(val_files, config, scaler, feature_cols, label_col)
    test_set = LazyTimeSeriesDataset(test_files, config, scaler, feature_cols, label_col)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    # 4. 初始化模型
    model = PatchFormer(
        seq_len=config.seq_len, patch_len=config.patch_len, pred_len=config.pre_len,
        enc_in=len(feature_cols), d_model=config.d_model, n_heads=8, e_layers=config.e_layers, dropout=config.drop_out
    ).to(device)

   # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
   # loss_fn = nn.MSELoss()
   # loss_fn = nn.HuberLoss(delta=0.5)
    #loss_fn = nn.HuberLoss(delta=1.0)
    loss_fn = nn.L1Loss()
    # 当 val_loss 在 3 个 epoch 内不下降时，学习率乘以 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 5. 训练循环
    train_loss_history, val_loss_history = [], []
    best_val_loss = float('inf')
    early_stop_count = 0

    patience_early_stop = 15

    print("开始训练...")
    for epoch in range(config.epochs):
        model.train()
        t_losses = []
        for seq, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            seq, label = seq.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(seq).reshape(label.shape)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())

        # 验证步
        model.eval()
        v_losses = []
        with torch.no_grad():
            for seq, label in val_loader:
                seq, label = seq.to(device), label.to(device)
                pred = model(seq).reshape(label.shape)
                v_losses.append(loss_fn(pred, label).item())

        avg_t, avg_v = np.mean(t_losses), np.mean(v_losses)
        train_loss_history.append(avg_t)
        val_loss_history.append(avg_v)
        # 更新学习率 (基于验证集 Loss)
        scheduler.step(avg_v)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_t:.6f} | Val Loss: {avg_v:.6f}")

        # 早停与最佳模型保存
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(model.state_dict(), "best_model.pth")
            early_stop_count = 0
            print("  --> 验证集表现提升，模型已保存。")
        else:
            early_stop_count += 1
            if early_stop_count >= patience_early_stop:
                print(f"!!! 早停触发：验证集 Loss 已连续 {patience_early_stop} 个 Epoch 未改善。")
                break

    # 绘制 Loss 曲线
    plot_loss_data(train_loss_history, val_loss_history)

    # 6. 测试评价
    # 6. 测试评价 (整合版：保存详细文件 + 计算总指标)
    print("开始模型测试（按文件对齐 + 统计总指标）...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    os.makedirs("Test_Detailed_Results", exist_ok=True)
    os.makedirs("Test_Error_Tracking_Figures", exist_ok=True)

    # 用于统计总指标的列表
    final_all_preds = []
    final_all_labels = []

    with torch.no_grad():
        for f_path in tqdm(test_files, desc="Processing Test Files"):
            df_test = pd.read_csv(f_path, encoding='gbk')

            # --- 预处理与训练一致 ---
          #  df_test['I_diff'] = df_test['Enhanced_Current'].diff().fillna(0)
          #  df_test[feature_cols] = df_test[feature_cols].rolling(window=3, min_periods=1).mean()
            df_test[label_col] = df_test[label_col].rolling(window=5, min_periods=1).mean()
            data_raw = df_test[feature_cols + label_col].fillna(0).values
            data_scaled = scaler.transform(data_raw)

            f_preds = []
            f_trues = []

            for i in range(len(data_scaled) - config.seq_len - config.pre_len + 1):
                seq = torch.FloatTensor(data_scaled[i: i + config.seq_len, :len(feature_cols)]).unsqueeze(0).to(device)
                pred = model(seq)

                p = scaler.inverse_transform_label(pred.cpu().numpy().flatten())
                l = scaler.inverse_transform_label(data_scaled[i + config.seq_len, -1])

                f_preds.append(float(p))
                f_trues.append(float(l))

            # 统计到总列表中
            final_all_preds.extend(f_preds)
            final_all_labels.extend(f_trues)

            # --- 保存该文件的详细结果 (包含最终补偿后的 SOE) ---
            res_df = df_test.iloc[config.seq_len: config.seq_len + len(f_preds)].copy()
            res_df['Predict_Error'] = f_preds
            res_df['True_Error'] = f_trues

            # 【核心公式】最终预测 SOE = ASWEKF 结果 - 预测出的误差
            res_df['Compensated_SOE'] = res_df['ASWEKF_SOE'] - res_df['Predict_Error']
            res_df['Compensated_SOE'] = res_df['Compensated_SOE'].clip(0, 100)
         #   res_df['Predicted_SOE'] = f_preds
          #  res_df['True_SOE'] = f_trues
         #   res_df['Predicted_SOE'] = res_df['Predicted_SOE'].clip(0, 100)  # 物理限幅

            f_name = os.path.basename(f_path)
            res_df.to_csv(f"Test_Detailed_Results/Res_{f_name}", index=False)
            # --- 为每一个测试放电片段单独绘制误差预测轨迹图 ---
            if len(f_preds) > 0:
                LABEL_FONT_SIZE = 12
                TICK_FONT_SIZE = 10.5
                LEGEND_FONT_SIZE = 10.5

                fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)

                plot_len = len(f_trues)

                # 构造真实时间轴：采样间隔为 10 s
                time_s = np.arange(plot_len) * 10

                ax.plot(
                    time_s,
                    f_trues,
                    label='True',
                    color='black',
                    linewidth=1.6,
                    alpha=0.85
                )

                ax.plot(
                    time_s,
                    f_preds,
                    label='Predicted',
                    color='red',
                    linewidth=1.4,
                    alpha=0.90
                )

                ax.set_xlabel('Time (s)', fontsize=LABEL_FONT_SIZE)

                # 强制横轴显示到 10000 s
                ax.set_xlim(0, 10000)
                ax.set_xticks(np.arange(0, 10001, 2000))
                ax.set_ylabel('SOE Error (%)', fontsize=LABEL_FONT_SIZE)
                ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                ax.legend(
                    loc='upper right',
                    bbox_to_anchor=(0.99, 1.00),
                    fontsize=LEGEND_FONT_SIZE,
                    frameon=False,
                    borderaxespad=0.0
                )

                ax.grid(False)
                plt.tight_layout()

                # 文件名去掉 .csv 后缀，避免保存图片名太长或重复
                fig_name = os.path.splitext(f_name)[0]
                save_path = os.path.join(
                    "Test_Error_Tracking_Figures",
                    f"Error_Tracking_{fig_name}.png"
                )

                plt.savefig(save_path, dpi=300)
                plt.close(fig)

    # --- 7. 计算并打印总指标 ---
    final_all_preds = np.array(final_all_preds)
    final_all_labels = np.array(final_all_labels)

    mae, rmse, mape, me = calculate_metrics(final_all_labels, final_all_preds)

    print(f"\n--- 测试集最终评价指标 (Error 预测值) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"ME:   {me:.4f}")

    # --- 8. 绘图 (前 1000 个采样点的 Error 拟合情况) ---
    # --- 8. 绘图：真实误差与 PatchFormer 预测误差对比 ---
    LABEL_FONT_SIZE = 12
    TICK_FONT_SIZE = 10.5
    LEGEND_FONT_SIZE = 10.5

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)

    # 前 1000 个点，采样间隔为 10 s
    plot_len = 1000
    time_s = np.arange(plot_len) * 10

    ax.plot(
        time_s,
        final_all_labels[:1000],
        label='True',
        color='black',
        linewidth=1.6,
        alpha=0.85
    )

    ax.plot(
        time_s,
        final_all_preds[:1000],
        label='Predicted',
        color='red',
        linewidth=1.4,
        alpha=0.90
    )

    ax.set_xlabel('Time (s)', fontsize=LABEL_FONT_SIZE)

    # 1000 个点对应 0–9990 s，这里显示到 10000 s 更清楚
    ax.set_xlim(0, 10000)
    ax.set_xticks(np.arange(0, 10001, 2000))
    ax.set_ylabel('SOE Error (%)', fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    # 统一 y 轴小数格式，可根据需要改成 %.2f
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(0.99, 1.00),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        borderaxespad=0.0
    )

    ax.grid(False)

    plt.tight_layout()

    plt.savefig(
        'PatchFormer_Error_Prediction_Tracking.png',
        dpi=300
    )

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str,
                        default=r'E:\SOE\实车数据集处理\切分片段结果\电车公司1_505Ah\Model_Features')
    parser.add_argument('-seq_len',
                        type=int, default=16)
    parser.add_argument('-patch_len', type=int, default=4)
    parser.add_argument('-pre_len', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-drop_out', type=float, default=0.5)  # 提高到0.4，增强抗噪能力

    # --- 新增模型结构参数 ---
    parser.add_argument('-d_model', type=int, default=32)
    parser.add_argument('-e_layers', type=int, default=1)

    main(parser.parse_args())