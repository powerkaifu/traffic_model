import numpy as np
import pandas as pd
import os
import json
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model  # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

# 自己的模組
from data_process import combine_vd_dataframes
from feature_engineering import prepare_features, inverse_transform_all
from predictor import build_model, train_model, evaluate_test, evaluate_train, predict_new
from visualizer import *
from shap_utils import explain_shap_feature

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


# 主程式
def main():
  base_dir = "."
  vd_folders = [ 'VLRJM60', 'VLRJX00', 'VLRJX20']
  # date_file = '2025-02-17_2025-02-23.json'
  # date_file = '2025-02-24_2025-03-02.json'
  # date_file = '2025-03-03_2025-03-09.json'
  # date_file = '2025-03-10_2025-03-16.json'
  # date_file = '2025-03-17_2025-03-23.json'
  # date_file = '2025-03-24_2025-03-30.json'
  # date_file = '2025-03-31_2025-04-06.json'
  date_file = '2025-05-05_2025-05-11.json'
  # date_file = '2025-06-02_2025-06-08.json'

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

  print("y 的形狀：", y.shape)  # 查看形狀（幾筆資料）
  print("y 的前幾筆資料：\n", y[: 10])  # 顯示前 10 筆
  print("y 的最小值：", np.min(y))
  print("y 的最大值：", np.max(y))

  # 確保是 float 類型
  X = X.astype(float)
  y = y.astype(float)

  # 分割訓練集與測試集 -------------ㄦ------------------------------------------------------------------------------------
  # X_train, y_train => 訓練集
  # X_test, y_test => 測試集
  X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      test_size=0.2,
      random_state=42,
      shuffle=False,
  )

  print("🔎 訓練集綠燈秒數區間")
  print("最小值：", y_train.min())
  print("最大值：", y_train.max())

  print("\n🔎 測試集綠燈秒數區間")
  print("最小值：", y_test.min())
  print("最大值：", y_test.max())

  # 查看 y_train 的分布情形
  print("📊 y_train 統計摘要：")
  print(pd.Series(y_train.flatten()).describe())

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
  # model.save(model_path)
  # print(f"✅ 模型已儲存到 {model_path}")

  # 儲存 scaler 標準化器 ----------------------------------------------------------------------------------------------
  if scaler is not None:
    joblib.dump(scaler, scaler_path)
    print(f"✅ 特徵縮放器已儲存到 {scaler_path}")
  else:
    print("⚠️ scaler 為空，無法儲存")

  # 模型評估 ---------------------------------------------------------------------------------------------------------
  print("📊 開始模型評估...評估訓練集、測試集")
  y_pred_train = evaluate_train(model, X_train, y_train)
  y_pred_test = evaluate_test(model, X_test, y_test)
  # batch_predictions_scaled = predict_new(model, X_test)
  print("✅ 模型評估完成。")

  # 繪製圖表 ---------------------------------------------------------------------------------------------------------
  # 合併資料的資料視覺化
  print(merged_df.columns)
  # plot_feature_distributions(merged_df, [ "Occupancy", "Speed", "Volume"])

  # merged_df['hour'] = pd.to_datetime(merged_df['timestamp']).dt.hour
  # plot_hourly_distributions(merged_df, [ "Occupancy", "Speed", "Volume_S"])

  # 反標準化 Occupancy，這才能顯示原本的佔用率（因為標準化後的 Occupancy 會落在 -n 到 +n 之間）
  df_viz = df.copy()
  features_to_inverse = list(scaler.feature_names_in_)
  df_viz = inverse_transform_all(df, scaler, features_to_inverse)
  # plot_occupancy_time_trend(df_viz)

  ## 散點圖-訓練集與測試集
  fig, axs = plt.subplots(1, 2, figsize = ( 12, 5 ))  # 一排兩張圖
  plot_scatter_predictions(y_train.flatten(), y_pred_train.flatten(), ax = axs[0], title = "訓練集散點圖")
  plot_scatter_predictions(y_test.flatten(), y_pred_test.flatten(), ax = axs[1], title = "測試集散點圖")
  plt.tight_layout()
  plt.show()

  print("訓練集預測最小綠燈秒數：", np.min(y_pred_train))
  print("訓練集預測最大綠燈秒數：", np.max(y_pred_train))
  print("測試集預測最小綠燈秒數：", np.min(y_pred_test))
  print("測試集預測最大綠燈秒數：", np.max(y_pred_test))

  # 誤差分布圖
  # plot_residuals(y_test, y_pred_test)

  # SHAP
  # explain_shap_feature(model, X_train, X_test, feature_names, output_dir = "shap")


# 執行主程式
if __name__ == "__main__":
  main()
