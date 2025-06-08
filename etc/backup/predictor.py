# predictor.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import Lambda  # type: ignore
from keras.layers import Rescaling  # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from visualizer import plot_scatter_predictions


# 建立模型
# input_shape 是模型的輸入形狀，通常是特徵數量，目前 26。
def build_model(input_shape):
  model = Sequential([ # Sequential() 是 Keras 中的一種模型類型，表示一個線性堆疊的神經網路。
      Input(shape = ( input_shape,)),  # 輸入層，告訴模型輸入的特徵數量
      Dense(64, activation = 'relu'),  # 第一層隱藏層，有 64 個神經元（節點），每個神經元會執行一個簡單的計算，並用 ReLU 非線性激活函數來讓模型能學習複雜的資料模式
      Dense(32, activation = 'relu'),  # 第二層隱藏層，有 32 個神經元，功能同上
      Dense(1),  # 輸出層，產生預測值，只有一個神經元(連續數值->綠燈秒數)
      # Rescaling(scale=79.0, offset=20.0)  # Rescaling 層，用來將輸出值縮放到 20 到 99 秒之間
  ])
  # optimizer=Adam 是優化器，用來更新模型權重，讓損失函數（誤差）變小
  # learning_rate=0.001 是更新的速度，太大可能不穩定，太小學得慢
  # loss='mse' 是損失函數，用「均方誤差」衡量預測值和真實值差距
  model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mse')
  return model


# 訓練模型
def train_model(model, X_train, y_train, epochs = 50):
  # EarlyStopping 回調函數，用來在訓練過程中監控模型的表現，防止過擬合。
  early_stop = EarlyStopping(
      monitor = 'val_loss',  # 驗證集的損失值，代表模型在沒看過的資料上的預測誤差。
      patience = 5,  # 連續 5 次訓練輪次都沒明顯進步，就提前停止訓練。
      min_delta = 0.001,  # ✅ 設定改善幅度，過小進步也不算
      restore_best_weights = True  # 回復到最佳模型參數
  )

  # 開始訓練模型
  model.fit(
      X_train,  # 訓練資料_特徵
      y_train,  # 訓練資料_目標（秒數）
      epochs = epochs,  # 訓練輪數
      batch_size = 32,  # 批次大小，每次訓練用 32 筆資料做一個小更新。
      validation_split = 0.2,  # 從 X_train/y_train 中切出 20% 當作驗證集，即時監控是否過擬合
      callbacks = [early_stop],  # 啟用早停機制監控訓練過程。
      verbose = 1  # 印出訓練過程資訊（每輪損失、驗證損失等）
  )


# 評估模型
def evaluate_train(model, X_train, y_train):
  y_pred = model.predict(X_train)

  mse = mean_squared_error(y_train, y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_train, y_pred)
  r2 = r2_score(y_train, y_pred)

  print("訓練集模型評估結果：")
  print(f"📉 MSE（均方誤差）: {mse:.2f}")
  print(f"📉 RMSE（均方根誤差）: {rmse:.2f}")
  print(f"📉 MAE（平均絕對誤差）: {mae:.2f}")
  print(f"📈 R²（決定係數）: {r2:.4f}")

  return y_pred


def evaluate_test(model, X_test, y_test):
  y_pred = model.predict(X_test)

  mse = mean_squared_error(y_test, y_pred)  # 均方誤差
  rmse = np.sqrt(mse)  # 均方根誤差
  mae = mean_absolute_error(y_test, y_pred)  # 平均絕對誤差
  r2 = r2_score(y_test, y_pred)  # 決定係數

  print("測試集模型評估結果：")
  print(f"📉 MSE（均方誤差）: {mse:.2f}")
  print(f"📉 RMSE（均方根誤差）: {rmse:.2f}")
  print(f"📉 MAE（平均絕對誤差）: {mae:.2f}")
  print(f"📈 R²（決定係數）: {r2:.4f}")

  return y_pred


def evaluate_train(model, X_train, y_train):
  y_pred = model.predict(X_train)

  mse = mean_squared_error(y_train, y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_train, y_pred)
  r2 = r2_score(y_train, y_pred)

  print("訓練集模型評估結果：")
  print(f"📉 MSE（均方誤差）: {mse:.2f}")
  print(f"📉 RMSE（均方根誤差）: {rmse:.2f}")
  print(f"📉 MAE（平均絕對誤差）: {mae:.2f}")
  print(f"📈 R²（決定係數）: {r2:.4f}")

  return y_pred


# 預測新資料
# def predict_new(model, X_new):
#   predictions = model.predict(X_new)
#   return np.round(predictions)  # 綠燈秒數通常是整數


def predict_new(model, X_new, min_sec = 20, max_sec = 99):
  predictions = model.predict(X_new)
  rounded = np.round(predictions).astype(int)  # 四捨五入成整數
  clipped = np.clip(rounded, min_sec, max_sec)  # 限制範圍在20~99
  return clipped
