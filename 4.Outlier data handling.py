import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os
from glob import glob


def anomaly_detection_vote(df, column_name):
    """对指定的列进行三种算法检测并投票"""
    data_series = df[column_name].ffill().bfill()
    if data_series.nunique() <= 1:  # 如果数据全是同一个值，跳过检测
        return pd.Series([False] * len(df))

    X = data_series.values.reshape(-1, 1)

    # 1. 箱型图
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1
    box_outliers = (data_series < (Q1 - 1.5 * IQR)) | (data_series > (Q3 + 1.5 * IQR))

    # 2. K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_scaled)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    distances = np.sqrt(((X_scaled - centroids[labels]) ** 2).sum(axis=1))
    km_outliers = distances > (np.mean(distances) + np.std(distances))

    # 3. 孤立森林
    clf = IsolationForest(contamination=0.05, random_state=42)
    if_labels = clf.fit_predict(X)
    if_outliers = if_labels == -1

    # 投票 (三选二)
    vote_count = box_outliers.astype(int) + km_outliers.astype(int) + if_outliers.astype(int)
    return vote_count >= 2


def process_all_companies_anomaly(root_dir):
    """
    遍历根目录下所有公司、所有类型的片段进行异常值处理
    """
    # 使用 ** 匹配所有子文件夹下的 csv
    # 路径示例：根目录/公司1/充电片段/*.csv
    file_list = glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)

    if not file_list:
        print("未找到CSV文件，请检查路径。")
        return

    target_cols = ['电池单体电压最高值', '电池单体电压最低值', '总电压', 'SOC', '总电流']

    print(f"开始处理，共发现 {len(file_list)} 个文件...")

    for file_path in file_list:
        # 获取文件名和所属分类（用于打印显示）
        rel_path = os.path.relpath(file_path, root_dir)
        print(f"正在清洗: {rel_path}")

        try:
            df = pd.read_csv(file_path)
            modified = False

            for col in target_cols:
                if col in df.columns:
                    is_anomaly = anomaly_detection_vote(df, col)
                    num_anomalies = is_anomaly.sum()

                    if num_anomalies > 0:
                        df.loc[is_anomaly, col] = np.nan
                        # 线性插值修复
                        df[col] = df[col].interpolate(method='linear').ffill().bfill()
                        modified = True

            if modified:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"处理文件 {rel_path} 时出错: {e}")

    print("\n--- 所有片段异常值清洗完成 ---")


# --- 路径配置 ---
# 这里指向你上一阶段“切分片段结果”的根目录
output_root = r"E:\SOE\实车数据集处理\切分片段结果"

process_all_companies_anomaly(output_root)