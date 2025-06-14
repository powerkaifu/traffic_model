# feature_engineering.py
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  # 導入 joblib 用於儲存/載入模型和 scaler

from webster import assign_green_seconds


# 特徵工程
def prepare_features(df, is_training = True, return_indices = False):
  try:
    # 1. 特徵選擇：挑選需要用的欄位，保留 timestamp 用於後續時間特徵擷取
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

    # 保存原始 DataFrame 的索引，方便後續追蹤資料來源
    original_indices = df.index

    # 2. 時間特徵提取：從 timestamp 拆出時、星期幾、分、秒，並建立是否尖峰時段欄位
    df['Hour'] = df['timestamp'].dt.hour
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek
    df['Minute'] = df['timestamp'].dt.minute
    df['Second'] = df['timestamp'].dt.second
    df['IsPeakHour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 17) & (df['Hour'] <= 19))
    df['IsPeakHour'] = df['IsPeakHour'].astype(int)  # 轉換為整數 (0 或 1)
    df = df.drop(columns = ['timestamp'])  # 刪除原始 timestamp 欄位

    # 3. 類別特徵處理 (One-Hot Encoding)：將 VD_ID 編碼成多個欄位，方便模型學習
    # 為了確保訓練和預測時的 One-Hot 編碼欄位一致，
    # 這裡需要一個固定的VD_ID列表。通常這會從訓練數據中收集。
    # 為了這個範例，我們先假設會有 'VLRJM60', 'VLRJX00', 'VLRJX20'
    # 實際應用中，你可能需要將所有可能的VD_ID保存下來。
    all_vd_ids = [ 'VLRJM60', 'VLRJX00', 'VLRJX20']  # 假設所有可能的 VD_ID
    df = pd.get_dummies(df, columns = ['VD_ID'], prefix = ['VD_ID'])
    # 確保 One-Hot 編碼後的欄位與訓練時一致
    for vd_id_col in [ f'VD_ID_{id}' for id in all_vd_ids ]:
      if vd_id_col not in df.columns:
        df[vd_id_col] = 0  # 補上缺失的 VD_ID 欄位為 0

    # 4. 資料標準化 (StandardScaler)：對數值型特徵做標準化，讓不同單位的特徵尺度一致
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
        'Hour',  # 小時
        'Minute',  # 分鐘
        'Second',  # 秒鐘
        'LaneID',  # 車道編號
        'LaneType',  # 車道類型
        'IsPeakHour'  # 是否尖峰時段 (0/1)
    ]
    # 注意：需要先確保所有 numerical_features 都存在於 df 中
    # 如果某些特徵在某次執行中沒有，可能會出錯。
    # 你可能需要增加一個檢查，例如：
    numerical_features = [ f for f in numerical_features if f in df.columns ]

    # 取得所有特徵名稱，包括 One-Hot 編碼後的 VD_ID 欄位
    # 這裡先收集，因為 scaler.fit_transform 會改變 df[numerical_features] 的值，
    # 但不會影響欄位名稱。
    all_current_features = df.columns.tolist()

    scaler = StandardScaler()  # 標準化（Standardization)，初始化標準化器
    # shap: 標準化後的數值型特徵會對應 SHAP 分析中的特徵值，確保尺度一致

    # 對數值特徵進行標準化（Z-score），轉換成有 ± 值的資料分佈（均值 0, 標準差 1）
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # ⭐️ 關鍵修改：在訓練階段儲存 StandardScaler
    if is_training:
      # 確保有目錄來保存 scaler
      os.makedirs('./traffic_models', exist_ok = True)
      joblib.dump(scaler, './traffic_models/scaler.pkl')
      print("✅ StandardScaler 已保存至 './traffic_models/scaler.pkl'")

    # 5. 準備特徵矩陣 (X) 和目標變數 (y)
    if is_training:
      # 如果是訓練階段，呼叫 assign_green_seconds 計算目標綠燈秒數
      df = assign_green_seconds(df)  # 注意：這裡的 df 應該包含所有需要計算 green_seconds 的特徵
      # 從 df 中排除 'green_seconds' 來獲取最終的特徵名稱
      feature_names = [ col for col in df.columns if col != 'green_seconds']
      # Pandas.values 提供轉換為 NumPy 陣列(ndarray)
      X = df[feature_names].values  # 特徵矩陣 (確保使用正確的欄位順序)
      y = df['green_seconds'].values.reshape(-1, 1)  # 目標變數 (綠燈秒數)

      if return_indices:
        # 回傳 X: 特徵矩陣, y: 目標變數, original_indices: 原始索引, feature_names: 特徵名稱
        return X, y, original_indices, feature_names, df
      else:
        return X, y, feature_names, df
    else:
      # 非訓練階段只需回傳特徵矩陣
      # 在這裡，X 應該是已經經過 One-Hot 和縮放的 DataFrame
      X = df[all_current_features].values  # 使用訓練時確立的特徵順序
      if return_indices:
        return X, original_indices, all_current_features, df  # 返回所有特徵名稱
      else:
        return X, all_current_features, df  # 返回所有特徵名稱

  except Exception as e:
    print(f"preprocess_data 發生錯誤：{e}")
    import traceback
    traceback.print_exc()  # 打印詳細錯誤堆疊
    return None, None, None, None, None  # 確保所有回傳值都為 None


# 單一特徵逆標準化，轉換回原始值
def inverse_transform_feature(df, feature_name, scaler):
  idx = list(scaler.feature_names_in_).index(feature_name)
  mean = scaler.mean_[idx]
  scale = scaler.scale_[idx]
  return df[feature_name] * scale + mean


# 多個特徵批次反標準化
def inverse_transform_all(df, scaler, features):
  df_copy = df.copy()
  for feature in features:
    df_copy[feature] = inverse_transform_feature(df, feature, scaler)
  return df_copy
