import pandas as pd
import os
import json
import joblib
import numpy as np  # 新增導入 numpy
from tensorflow.keras.models import load_model  # type: ignore

# 自己的模組
# from predictor import predict_new # 假設你的 predict_new 會返回原始預測值


# 定義一個帶有裁剪邏輯的預測函數
def predict_with_clipping(model, X_new_data, min_val = 30.0, max_val = 99.0):
  """
    使用模型進行預測，並將結果裁剪到指定範圍內。
    """
  predicted_green_seconds_raw = model.predict(X_new_data)

  # 對每個預測值進行裁剪
  clipped_green_seconds = np.clip(predicted_green_seconds_raw, min_val, max_val)

  # 四捨五入並轉換為整數
  final_green_seconds = np.round(clipped_green_seconds).astype(int)

  return final_green_seconds


# 你的 preprocess_and_scale_new_data 函數 (保持不變)
def preprocess_and_scale_new_data(new_data_df: pd.DataFrame, feature_names: list, scaler):
  one_hot_vd_cols = [ col for col in feature_names if col.startswith('VD_ID_') ]
  new_data_df_processed = pd.get_dummies(new_data_df, columns = ['VD_ID'], prefix = 'VD_ID')

  # 補齊缺少的 one-hot 欄位
  for col in one_hot_vd_cols:
    if col not in new_data_df_processed.columns:
      new_data_df_processed[col] = 0

  numerical_features_to_scale = [
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
      'DayOfWeek',
      'Hour',
      'Minute',
      'Second',
      'LaneID',
      'LaneType',
      'IsPeakHour',
  ]
  existing_numerical = [ f for f in numerical_features_to_scale if f in new_data_df_processed.columns ]
  new_data_df_processed[existing_numerical] = new_data_df_processed[existing_numerical].astype(float)

  # 標準化
  scaled_values = scaler.transform(new_data_df_processed[existing_numerical])
  scaled_df = pd.DataFrame(scaled_values, columns = existing_numerical, index = new_data_df_processed.index)
  new_data_df_processed.loc[:, existing_numerical] = scaled_df

  # 補齊欄位順序並填 0
  X_new_final = pd.DataFrame(0, index = new_data_df_processed.index, columns = feature_names)
  for col in feature_names:
    if col in new_data_df_processed.columns:
      X_new_final[col] = new_data_df_processed[col]

  return X_new_final.values.astype(float)


# 🔧 特徵順序（與訓練模型一致）
feature_names = [
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
    'Hour',
    'DayOfWeek',
    'Minute',
    'Second',
    'IsPeakHour',
    'VD_ID_VLRJM60',
    'VD_ID_VLRJX00',
    'VD_ID_VLRJX20',
    'Occ_x_Volume_S',
    'Occ_x_Volume_L',
    'Occ_x_Volume_T',
    'SpeedS_x_VolumeS',
    'SpeedL_x_VolumeL',
    'SpeedT_x_VolumeT'
]

# 📂 路徑設定
model_path = './traffic_models/trained_model.keras'
scaler_path = './traffic_models/scaler.pkl'
samples_path = 'sample_inputs.json'

# ✅ 載入模型與 scaler
model = load_model(model_path) if os.path.exists(model_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# 🚦 批次預測流程
if not all([ model, scaler ]) or not os.path.exists(samples_path):
  print("❌ 請確認模型、特徵縮放器與 sample_inputs.json 都存在！")
else:
  print("📌 批次預測資料結果：")
  with open(samples_path, 'r') as f:
    samples_inputs = json.load(f)

  for i, sample_input in enumerate(samples_inputs):
    df = pd.DataFrame([sample_input])
    X_new = preprocess_and_scale_new_data(df, feature_names, scaler)

    # 這裡調用新的預測函數，它會處理裁剪
    pred_clipped = predict_with_clipping(model, X_new, min_val = 20.0, max_val = 99.0)

    hour = sample_input.get('Hour', -1)
    minute = sample_input.get('Minute', -1)
    is_peak = sample_input.get('IsPeakHour', 0)
    peak_status = "尖峰" if is_peak == 1 else "離峰"

    # pred_clipped 的形狀是 (1, 1)，所以取值是 pred_clipped[0][0]
    print(f"範例 {i+1}：時間 {hour:02d}:{minute:02d}（{peak_status}） → 預測綠燈秒數 = {pred_clipped[0][0]:.0f} 秒")  # 改為 .0f 顯示整數
