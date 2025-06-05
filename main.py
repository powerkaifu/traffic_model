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

# 主程式
def main():
  base_dir = "."
  vd_folders = [ 'VLRJM60', 'VLRJX00', 'VLRJX20']
  date_file = '2025-05-05_2025-05-11.json'

  # 合併多個偵測器資料為一個 DataFrame
  merged_df = combine_vd_dataframes(base_dir, vd_folders, date_file)
  if merged_df is None:
    print("❌ 沒有讀取到任何 VD 的資料。")
    return

  print(f"合併後的 DataFrame 資料筆數：{len(merged_df)}")
  print(f"合併後的 DataFrame 欄位：{merged_df.columns}")
  print("-" * 80)

  # 進行特徵工程 --------------------------------------------------------------------------------------------------------
  ## 一般特徵標準化的 X 數據都落於 -3 到 3 之間
  X, y, original_indices, feature_names, df = prepare_features(merged_df, is_training = True, return_indices = True)
  print(feature_names)
  if X is None or y is None or feature_names is None:
    print("❌ 資料前處理失敗，程式終止。")
    return

  # 確保是 float 類型
  X = X.astype(float)
  y = y.astype(float)

  # 分割訓練集與測試集 -------------ㄦ------------------------------------------------------------------------------------
  use_split = True  # True 代表用訓練測試分割，False 代表全部訓練
  if use_split:
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

  # 載入模型，否則建立新模型 ----------------------------------------------------------------------------------------------
  model_path = './traffic_models/trained_model.keras'  # 模型儲存路徑
  scaler_path = './traffic_models/scaler.pkl'  # 特徵縮放器儲存路徑
  if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ 已加載先前訓練的模型")
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mse')
  else:
    print("⚠️ 模型檔案不存在，建立新的模型")
    print(f"⚠️ 建立新模型，輸入特徵數量：{X.shape[1]}")  # 26 個特徵
    model = build_model(input_shape = X.shape[1])
    print("🆕 建立新的模型")

  # 載入 scaler 用於標準化 --------------------------------------------------------------------------------------------
  scaler = None
  if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("✅ 已加載先前使用的特徵縮放器")
  else:
    if os.path.exists(model_path):
      print("⚠️ 載入特徵縮放器失敗：scaler.pkl 檔案不存在。")

  # 訓練模型 ---------------------------------------------------------------------------------------------------------
  print("⏳ 開始模型訓練...")
  train_model(model, X_train, y_train, epochs = 50)
  print("✅ 模型訓練完成。")

  # 儲存模型 ---------------------------------------------------------------------------------------------------------
  model.save(model_path)
  print(f"✅ 模型已儲存到 {model_path}")

  # 儲存 scaler 標準化器 ----------------------------------------------------------------------------------------------
  if scaler is not None:
    joblib.dump(scaler, scaler_path)
    print(f"✅ 特徵縮放器已儲存到 {scaler_path}")
  else:
    print("⚠️ scaler 為空，無法儲存")

  # 模型評估 ---------------------------------------------------------------------------------------------------------
  print("📊 開始模型評估...")
  y_pred = evaluate_model(model, X_test, y_test)
  batch_predictions_scaled = predict_new(model, X_test)
  print("✅ 模型評估完成。")

  # 繪製圖表 ---------------------------------------------------------------------------------------------------------
  ## 散點圖
  plot_scatter_predictions(y_test.flatten(), y_pred.flatten())

  # 反標準化 Occupancy，這才能顯示原本的佔用率（因為標準化後的 Occupancy 會落在 -n 到 +n 之間）
  df_viz = df.copy()
  features_to_inverse = list(scaler.feature_names_in_)
  df_viz = inverse_transform_all(df, scaler, features_to_inverse)
  # plot_occupancy_vs_green_seconds(df_viz)
  # plot_occupancy_distribution(df_viz)
  # plot_occupancy_time_trend(df_viz)

  # plot_volume_distribution(merged_df)
  # plot_speed_distribution(merged_df)
  # plot_residuals(y_test, y_pred)

  # SHAP
  # print('-' * 80)
  # explain_shap_feature(model, X_train, X_test, feature_names, output_dir = "shap")
  # print('-' * 80)

# 執行主程式
if __name__ == "__main__":
  main()
