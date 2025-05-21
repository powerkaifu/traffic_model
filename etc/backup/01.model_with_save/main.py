import pandas as pd
from sklearn.model_selection import train_test_split
from data_process import combine_vd_dataframes, preprocess_data
from predictor import build_model, train_model, evaluate_model, predict_new
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import os

# print(tf.__version__) # 2.19.0
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# 設定 VD 資料夾和檔案名稱
base_dir = "."
vd_folders = [ 'VLRJM60', 'VLRJX00', 'VLRJX20']
# Week 1
# date_file = '2025-02-17_2025-02-23.json'
# Week 2
# date_file = '2025-02-24_2025-03-02.json'
# Week 3
# date_file = '2025-03-03_2025-03-09.json'
# Week 4
# date_file = '2025-03-10_2025-03-16.json'
# Week 5
# date_file = '2025-03-17_2025-03-23.json'
# Week 6
# date_file = '2025-03-24_2025-03-30.json'
# Week 7
date_file = '2025-03-31_2025-04-06.json'

# 讀取並合併 VD 資料
merged_df = combine_vd_dataframes(base_dir, vd_folders, date_file)
print(f"合併後的 DataFrame 總欄位筆數：{len(merged_df)}")

if merged_df is not None:
  print("合併後的 DataFrame 資料欄位：")
  print(merged_df.head(1))
  print("================================================================================")

  # 回傳 X 和 y
  X, y = preprocess_data(merged_df)
  print("訓練資料：", X[0])
  print("================================================================================")

  # 轉換型別
  X = X.astype(float)
  y = y.astype(float)
  print("查看 X 和 y 的資料型態：")
  print(type(X), X.dtype)
  print(type(y), y.dtype)
  print("X 特徵資料集：")
  print(X[0])
  print("y 目標變數資料集：")
  print(y[0])
  print("================================================================================")

  # 分割訓練與測試集
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

  # 嘗試載入模型；若失敗則建立新模型
  # model_path = 'trained_model.h5'
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
    model = build_model(input_shape = X.shape[1])
    print("🆕 建立新的模型")

  # 使用新資料繼續訓練模型
  train_model(model, X_train, y_train, epochs = 50)

  # 儲存模型
  model.save(model_path)
  print("✅ 模型已保存")

  # 預測與篩選
  startIndex = 0
  endIndex = 4547
  over_seconds = 40
  y_pred = evaluate_model(model, X_test, y_test)
  new_pred = predict_new(model, X_test[startIndex : endIndex])
  print("新資料預測結果：")
  print(new_pred)
  print("================================================================================")

  print(f"預測綠燈秒數大於 {over_seconds} 秒的資料：")
  over = [(startIndex + i, pred[0]) for i, pred in enumerate(new_pred) if pred[0] > over_seconds]
  for idx, val in over:
    print(f"索引 {idx}，預測綠燈秒數：{val:.1f}")
else:
  print("❌ 沒有讀取到任何 VD 的資料。")
