import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_process import combine_vd_dataframes, preprocess_data
from predictor import build_model, train_model, evaluate_model, predict_new

pd.set_option('display.width', None)  # 不限制橫向寬度
pd.set_option('display.max_columns', None)  # 顯示所有欄位

# 設定 VD 資料夾和檔案名稱
base_dir = "."
vd_folders = [ 'VLRJM60', 'VLRJX00', 'VLRJX20']
date_file = '2025-02-17_2025-02-23.json'

# 讀取並合併 VD 資料
merged_df = combine_vd_dataframes(base_dir, vd_folders, date_file)
print(f"合併後的 DataFrame 總欄位筆數：{len(merged_df)}")

if merged_df is not None:
  print("合併後的 DataFrame 資料欄位：")
  print(merged_df.head(1))
  print("================================================================================")

  # 回傳 X 和 y，其中 X 為輸入特徵（features），y 為目標變數（labels or targets）
  X, y = preprocess_data(merged_df)

  print("訓練資料：", X[0])
  print("================================================================================")

  # 將 X, y 轉成浮點數，模型訓練只能使用數字
  # X, y 資料型態為
  X = X.astype(float)  # 特徵 X，例如車速、佔用率、各種車種的數量與速度、時間資訊等，是用來預測的條件。
  y = y.astype(float)  # 目標變數 y（例如綠燈秒數）
  print("查看 X 和 y 的資料型態：")
  print(type(X), X.dtype)
  print(type(y), y.dtype)
  print("X 特徵資料集：")
  print(X[0])
  print("y 目標變數資料集：")
  print(y[0])
  print("================================================================================")

  # 分割訓練，80% 訓練集，20% 測試集
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

  # 訓練模型
  # 設定模型輸入形狀
  model = build_model(input_shape = X.shape[1])
  train_model(model, X_train, y_train, epochs = 20)

  startIndex = 0
  endIndex = 10
  over_seconds = 30
  y_pred = evaluate_model(model, X_test, y_test)  # 測試集預測結果
  new_pred = predict_new(model, X_test[startIndex : endIndex])  # 預測幾筆到幾筆資料
  print("新資料預測結果：")
  print(new_pred)
  print("================================================================================")

  # 找出預測綠燈秒數大於 over_seconds 的資料
  print(f"預測綠燈秒數大於 {over_seconds} 秒的資料：")
  over = [(startIndex + i, pred[0]) for i, pred in enumerate(new_pred) if pred[0] > over_seconds]
  for idx, val in over:
    print(f"索引 {idx}，預測綠燈秒數：{val:.1f}")
else:
  print("沒有讀取到任何 VD 的資料。")
