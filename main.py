# main.py
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model  # type: ignore

from data_process import combine_vd_dataframes
from preprocess import preprocess_data
from predictor import build_model, train_model, evaluate_model, predict_new
from visualizer import plot_volume_distribution, plot_speed_distribution, plot_residuals

gpus = tf.config.list_physical_devices('GPU')
print('1111111111', tf.test.is_built_with_cuda())
print('222222222', tf.test.is_gpu_available())

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# 設定 VD 資料夾和檔案名稱
base_dir = "."
vd_folders = [ 'VLRJM60', 'VLRJX00', 'VLRJX20']
date_file = '2025-05-05_2025-05-11.json'
# date_file = 'test.json' # 測試用的檔案，希望模擬出特殊情況，例如天氣影響、交通事故的特徵資料

# 讀取並合併 VD 資料
merged_df = combine_vd_dataframes(base_dir, vd_folders, date_file)
print(f"合併後的 DataFrame 資料筆數：{len(merged_df)}")
print(f"合併後的 DataFrame 欄位：{merged_df.columns}")
# print(f"合併後的 DataFrame 前五筆：", merged_df.head())

# 繪圖-製流量和速度的分布圖 ------------------------------------------------------------------
# plot_volume_distribution(merged_df)
# plot_speed_distribution(merged_df)

# ----------------------------------------------------------------------------------------

if merged_df is not None:
  print("=" * 80)

  # 回傳 X, y 和原始索引
  X, y, original_indices = preprocess_data(merged_df, return_indices = True)

  # 資料轉型
  X = X.astype(float)
  y = y.astype(float)
  print("查看 X 和 y 的資料型態：")
  print(type(X), X.dtype, X.shape)
  print(type(y), y.dtype, y.shape)
  print("X 特徵資料集：", X[0])
  print("y 目標變數資料集：", y[0])
  print("=" * 80)

  # ✅ 控制是否使用訓練集 / 測試集分割 -------------------------------------------------------------------------------------------
  use_split = False  # True 使用分割；False 使用全部資料訓練

  if use_split:
    # 分割訓練與測試資料（含原始索引）
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, original_indices, test_size = 0.2, random_state = 42)
    print("📦 使用分割方式訓練")
  else:
    X_train, y_train = X, y
    X_test, y_test = X, y
    indices_train, indices_test = original_indices, original_indices
    print("📦 使用不分割方式訓練")

  # 嘗試載入模型，若失敗則建立新模型 ----------------------------------------------------------------------------------------------
  model_path = './traffic_models/trained_model.h5'
  try:
    if os.path.exists(model_path):
      model = load_model(model_path, custom_objects = { 'mse': tf.keras.losses.MeanSquaredError})
      print("✅ 已加載先前訓練的模型")
      model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mse')
    else:
      raise FileNotFoundError("模型檔案不存在")
  except Exception as e:
    print(f"⚠️ 載入模型失敗，原因：{e}")
    model = build_model(input_shape = X.shape[1])  # X.shape[1] 是特徵數量，20
    print("🆕 建立新的模型")

  # 模型訓練 -------------------------------------------------------------------------------------------------------------------
  train_model(model, X_train, y_train, epochs = 1)  # epochs 訓練的輪數

  # 儲存模型
  os.makedirs(os.path.dirname(model_path), exist_ok = True)
  model.save(model_path)
  print("✅ 模型已保存至:", model_path)

  # 預測與評估
  ## 如果用了不分割資料訓練，就不需要評估模型，因為沒有測試集，但好處就是可以用全部資料訓練模型
  y_pred = evaluate_model(model, X_test, y_test)  # y_pred 模型預測綠燈秒數的結果
  new_pred = predict_new(model, X_test)

  # 繪圖-殘差圖 ----------------------------------------------------------------------------------------------------------------
  # plot_residuals(y_test, y_pred)

  # 顯示預測結果，並輸出到 output.txt --------------------------------------------------------------------------------------------
  over_seconds = 50  # 預測綠燈秒數大於 50 秒
  print("=" * 80)
  print(f"預測綠燈秒數大於 {over_seconds} 秒的資料：")
  output_data_count = 100  # 限制輸出資料筆數
  over_predictions = [( i, pred[0] ) for i, pred in enumerate(new_pred) if pred[0] > over_seconds]

  with open("output.txt", "w", encoding = "utf-8") as f:
    for idx, ( test_index, val ) in enumerate(over_predictions):
      if idx >= output_data_count:
        break
      original_index_in_merged_df = indices_test[test_index]
      original_data = merged_df.loc[original_index_in_merged_df]
      # 寫入到 output.txt， file = f
      print(f"預測結果索引 {idx} (測試集索引: {test_index}, 原始 DataFrame 索引: {original_index_in_merged_df})，預測綠燈秒數：{val:.1f}", file = f)
      print("原始資料:", file = f)
      print(original_data, file = f)
      print("-" * 50, file = f)

else:
  print("❌ 沒有讀取到任何 VD 的資料。")
