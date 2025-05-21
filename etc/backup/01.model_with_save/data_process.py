import os
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler


# 資料處理：將 json 轉成 DataFrame
## json -> dict -> DataFrame
def json_to_dataframe(file_path):
  try:
    with open(file_path, 'r') as f:
      data = json.load(f)  # data 為 dict，key 為時間戳記，value 為車道資料
    # print(f"讀取檔案：{file_path}，資料筆數：{len(data)}")
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


# 資料處理：將每星期三支 VD 的資料合併為一個 DataFrame
def combine_vd_dataframes(base_dir, vd_folders, date_file):
  all_dfs = []
  for vd_folder in vd_folders:
    file_path = os.path.join(base_dir, vd_folder, date_file)  # 構建檔案路徑
    df = json_to_dataframe(file_path)  # 讀取 JSON 檔案轉換為 DataFrame
    if df is not None:
      df['VD_ID'] = vd_folder  # 新增 VD_ID 欄位
      all_dfs.append(df)
  if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index = True)  # 合併所有 DataFrame
    merged_df = merged_df.sort_values(by = 'timestamp').reset_index(drop = True)  # 重設索引
    return merged_df
  else:
    return None


# 模擬綠燈策略 01
# def assign_green_seconds(df):
#   base = df['Occupancy'].clip(upper = 60)
#   vehicle_bonus = (0.5 * df['Volume_S'] + 1.0 * df['Volume_L'] + 1.5 * df['Volume_T'])  # 車輛類型加成
#   speed_penalty = (df['Speed'] < 30).astype(int) * 10  # 車速懲罰：小於 30km/h 加 10 秒
#   peak_bonus = df['IsPeakHour'].astype(int) * 10  # 尖峰時段加成
#   return df


# 模擬綠燈策略 02
def assign_green_seconds(df):
  base = df['Occupancy'].clip(upper = 60)
  vehicle_bonus = (2 * df['Volume_S'] + 2.5 * df['Volume_L'] + 3.0 * df['Volume_T'])  # 車輛類型加成
  speed_penalty = (df['Speed'] < 30).astype(int) * 15  # 車速懲罰：小於 30km/h 加 10 秒
  peak_bonus = df['IsPeakHour'].astype(int) * 20  # 尖峰時段加成
  df['green_seconds'] = (base + vehicle_bonus + speed_penalty + peak_bonus).clip(20, 90).round()  # 綠燈秒數總計並限制區間
  return df


# 資料前處理
def preprocess_data(df, is_training = True):
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

    # 2. 時間特徵提取
    df['Hour'] = df['timestamp'].dt.hour
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek
    df['Minute'] = df['timestamp'].dt.minute
    df['Second'] = df['timestamp'].dt.second
    df['IsPeakHour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 17) & (df['Hour'] <= 19))
    df['IsPeakHour'] = df['IsPeakHour'].astype(int)  # 轉換為整數 (0 或 1)
    df = df.drop(columns = ['timestamp'])  # 刪除原始 timestamp

    # 3. 類別特徵處理 (One-Hot Encoding)
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
      return X, y
    else:
      X = df.values
      return X

  except Exception as e:
    print(f"preprocess_data 發生錯誤：{e}")
    return None
