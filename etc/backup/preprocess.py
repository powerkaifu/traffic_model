# preprocess.py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from webster import assign_green_seconds


# 資料前處理
def preprocess_data(df, is_training = True, return_indices = False):
  try:
    # 1. 特徵選擇
    selected_features = [
        'Speed',
        'Occupancy',
        'Volume_M',
        'Volume_S',
        'Volume_L',
        'Volume_T',
        'Speed_M',
        'Speed_S',
        'Speed_L',
        'Speed_T',
        'LaneID',
        'LaneType',
        'VD_ID',
        'timestamp'  # 保留 timestamp 用於時間特徵提取
    ]
    df = df[selected_features].copy()

    # 保存原始 DataFrame 的索引
    original_indices = df.index

    # 2. 時間特徵提取
    df['Hour'] = df['timestamp'].dt.hour
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek
    df['Minute'] = df['timestamp'].dt.minute
    df['Second'] = df['timestamp'].dt.second
    df['IsPeakHour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 17) & (df['Hour'] <= 19))
    df['IsPeakHour'] = df['IsPeakHour'].astype(int)  # 轉換為整數 (0 或 1)
    df = df.drop(columns = ['timestamp'])  # 刪除原始 timestamp

    # 3. 類別特徵處理 (One-Hot Encoding) - 將 VD_ID 轉換為數值，讓模型可以處理
    df = pd.get_dummies(df, columns = ['VD_ID'], prefix = ['VD_ID'])

    # 4. 資料縮放 (StandardScaler) - 全特徵輸入 16 個
    numerical_features = [
        'Speed',  # 平均速度
        'Occupancy',  # 車道佔用率
        'Volume_M',  # 機車流量
        'Volume_S',  # 小型車流量
        'Volume_L',  # 大型車流量
        'Volume_T',  # 聯結車流量
        'Speed_M',  # 機車平均速度
        'Speed_S',  # 小型車平均速度
        'Speed_L',  # 大型車平均速度
        'Speed_T',  # 聯結車平均速度
        'DayOfWeek',  # 星期幾
        'Hour',  # 幾時
        'Minute',  # 幾分
        'Second',  # 幾秒
        'LaneID',  # 車道編號
        'LaneType',  # 車道類型
    ]
    scaler = StandardScaler()  # 標準化數據
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # 5. 準備特徵矩陣 (X) 和目標變數 (y)
    if is_training:
      df = assign_green_seconds(df)
      y = df['green_seconds'].values.reshape(-1, 1)
      X = df.drop(columns = ['green_seconds']).values
      if return_indices:
        return X, y, original_indices
      else:
        return X, y
    else:
      X = df.values
      if return_indices:
        return X, original_indices
      else:
        return X

  except Exception as e:
    print(f"preprocess_data 發生錯誤：{e}")
    return None
