# data_process.py
import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
from webster import assign_green_seconds

# 資料處理：將 json 轉成 DataFrame
## json -> dict -> DataFrame
def json_to_dataframe(file_path):
  try:
    with open(file_path, 'r') as f:
      data = json.load(f)  # data 為 dict，key 為時間戳記，value 為車道資料
    rows = []
    for timestamp, lanes in data.items():
      for lane in lanes:
        row = { "timestamp": timestamp}  # 為每一筆新增一個 timestamp
        row.update(lane)  # update 是 dict 方法，可以更新 row，會有 timestamp 和車道資料
        for vehicle_type, vehicle_data in lane["Vehicles"].items():
          row[f"Volume_{vehicle_type}"] = vehicle_data["Volume"]
          row[f"Speed_{vehicle_type}"] = vehicle_data["Speed"]
        del row["Vehicles"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

  except FileNotFoundError:
    print(f"錯誤：找不到檔案於路徑 {file_path}")
    return None
  except Exception as e:
    print(f"發生錯誤：{e}")
    return None


# 資料處理：將每星期三支 VD 的資料合併為一個 DataFrame 來進行訓練
def combine_vd_dataframes(base_dir, vd_folders, date_file):
    linkid_to_direction = {
        '6004930400060A': 'South',
        '6004930000080A': 'North',
        '6001190200010A': 'East',
        '6001190600010A': 'West'
    }

    all_dfs = []
    for vd_folder in vd_folders:
        file_path = os.path.join(base_dir, vd_folder, date_file)
        df = json_to_dataframe(file_path)
        if df is not None:
            df['VD_ID'] = vd_folder

            # 新增 direction 欄位，依據 LinkID 映射
            df['Direction'] = df['LinkID'].map(linkid_to_direction)

            all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df = merged_df.sort_values(by='timestamp').reset_index(drop=True)
        return merged_df
    else:
        return None


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

    # 4. 資料縮放 (StandardScaler)
    numerical_features = [
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
        'Hour',
        'DayOfWeek',
        'Minute',
        'Second',
        'LaneID',
        'LaneType',
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
