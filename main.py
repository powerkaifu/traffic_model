import numpy as np
import pandas as pd
import os
import json
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model  # type: ignore

# 自己的模組
from data_process import combine_vd_dataframes
from feature_engineering import prepare_features, inverse_transform_all
from predictor import build_model, train_model, evaluate_model, predict_new
from visualizer import *
from shap_utils import explain_shap_feature

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


# 處理新的資料，將其轉換為模型可以接受的格式，並進行標準化
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
  numerical_features_to_scale_existing = [ f for f in numerical_features_to_scale if f in new_data_df_processed.columns ]

  # 先把欄位轉成 float，避免 dtype 不相容
  new_data_df_processed[numerical_features_to_scale_existing] = new_data_df_processed[numerical_features_to_scale_existing].astype(float)

  # 標準化數值特徵
  scaled_values = scaler.transform(new_data_df_processed[numerical_features_to_scale_existing])
  scaled_df = pd.DataFrame(scaled_values, columns = numerical_features_to_scale_existing, index = new_data_df_processed.index)
  new_data_df_processed.loc[:, numerical_features_to_scale_existing] = scaled_df

  # 補齊欄位順序並填零
  X_new_final_ordered = pd.DataFrame(0, index = new_data_df_processed.index, columns = feature_names)
  for col in feature_names:
    if col in new_data_df_processed.columns:
      X_new_final_ordered[col] = new_data_df_processed[col]

  return X_new_final_ordered.values.astype(float)


# 主程式
def main():
  base_dir = "."
  vd_folders = [ 'VLRJM60', 'VLRJX00', 'VLRJX20']
  date_file = '2025-02-17_2025-02-23.json'

  # 合併多個偵測器資料為一個 DataFrame
  merged_df = combine_vd_dataframes(base_dir, vd_folders, date_file)
  if merged_df is None:
    print("❌ 沒有讀取到任何 VD 的資料。")
    return

  print(f"合併後的 DataFrame 資料筆數：{len(merged_df)}")
  print(f"合併後的 DataFrame 欄位：{merged_df.columns}")
  print("-" * 80)

  # 進行特徵工程與目標變數分離
  X, y, original_indices, feature_names, df = prepare_features(merged_df, is_training = True, return_indices = True)

  if X is None or y is None or feature_names is None:
    print("❌ 資料前處理失敗，程式終止。")
    return

  X = X.astype(float)
  y = y.astype(float)

  use_split = True  # True 代表用訓練測試分割，False 代表全部訓練

  if use_split:
    # 分割成訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=False,
    )
    print("📦 使用分割方式訓練")
  else:
    X_train, y_train = X, y
    X_test, y_test = X, y
    print("📦 使用不分割方式訓練")

  model_path = './traffic_models/trained_model.keras'
  scaler_path = './traffic_models/scaler.pkl'

  # 載入已訓練模型，沒有則建立新模型
  if os.path.exists(model_path):
    model = load_model(model_path, custom_objects = { 'mse': tf.keras.losses.MeanSquaredError})
    print("✅ 已加載先前訓練的模型")
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mse')
  else:
    print("⚠️ 模型檔案不存在，建立新的模型")
    print(f"⚠️ 建立新模型，輸入特徵數量：{X.shape[1]}")  # 26 個特徵
    model = build_model(input_shape = X.shape[1])
    print("🆕 建立新的模型")

  # 載入 scaler 用於標準化
  scaler = None
  if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("✅ 已加載先前使用的特徵縮放器")
  else:
    if os.path.exists(model_path):
      print("⚠️ 載入特徵縮放器失敗：scaler.pkl 檔案不存在。")

  # 訓練模型
  print("⏳ 開始模型訓練...")
  train_model(model, X_train, y_train, epochs = 50)
  print("✅ 模型訓練完成。")

  # 儲存模型
  model.save(model_path)
  print(f"✅ 模型已儲存到 {model_path}")

  # 儲存 scaler 標準化器
  if scaler is not None:
    joblib.dump(scaler, scaler_path)
    print(f"✅ 特徵縮放器已儲存到 {scaler_path}")
  else:
    print("⚠️ scaler 為空，無法儲存")

  # 模型評估
  print("📊 開始模型評估...")
  y_pred = evaluate_model(model, X_test, y_test)
  batch_predictions_scaled = predict_new(model, X_test)
  print("✅ 模型評估完成。")

  # 畫圖範例（可以解除註解執行）
  # 反標準化 Occupancy，這樣才能顯示原本的佔用率
  df_viz = df.copy()
  features_to_inverse = list(scaler.feature_names_in_)
  df_viz = inverse_transform_all(df, scaler, features_to_inverse)
  plot_occupancy_vs_green_seconds(df_viz)
  plot_occupancy_distribution(df_viz)
  plot_occupancy_time_trend(df_viz)

  # plot_volume_distribution(merged_df)
  # plot_speed_distribution(merged_df)
  # plot_residuals(y_test, y_pred)

  # SHAP
  print('-' * 80)
  explain_shap_feature(model, X_train, X_test, feature_names, output_dir = "shap")
  print('-' * 80)

  # ✅ 從 JSON 讀取多筆資料並預測
  samples_path = 'sample_inputs.json'
  if os.path.exists(samples_path) and scaler is not None:
    print("📌 批次預測資料：")
    with open(samples_path, 'r') as f:
      samples_inputs = json.load(f)

      for i, sample_input in enumerate(samples_inputs):
        df = pd.DataFrame([sample_input])
        X_new = preprocess_and_scale_new_data(df, feature_names, scaler)
        pred = predict_new(model, X_new)

        # 顯示時間與是否尖峰
        hour = sample_input.get('Hour', -1)
        minute = sample_input.get('Minute', -1)
        is_peak = sample_input.get('IsPeakHour', 0)
        peak_status = "尖峰" if is_peak == 1 else "離峰"

        print(f"範例 {i+1}：時間 {hour:02d}:{minute:02d}（{peak_status}） → 預測綠燈秒數 = {pred[0][0]:.2f} 秒")
  else:
    print("⚠️ 無法執行批次預測，因為找不到 test_data.json 或特徵縮放器未載入。")


# 執行主程式
if __name__ == "__main__":
  main()
