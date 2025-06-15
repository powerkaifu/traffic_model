import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras import regularizers  # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========================
# ✅ 原始版本：保留註解完整備份
# ========================

# 建立模型
# input_shape 是模型的輸入形狀，通常是特徵數量，目前 26。
# def build_model(input_shape):
#   model = Sequential([ # Sequential() 是 Keras 中的一種模型類型，表示一個線性堆疊的神經網路。
#       Input(shape = ( input_shape,)),  # 輸入層，告訴模型輸入的特徵數量
#       Dense(64, activation = 'relu'),  # 第一層隱藏層，有 64 個神經元（節點），每個神經元會執行一個簡單的計算，並用 ReLU 非線性激活函數來讓模型能學習複雜的資料模式
#       Dense(32, activation = 'relu'),  # 第二層隱藏層，有 32 個神經元，功能同上
#       Dense(1),  # 輸出層，產生預測值，只有一個神經元(連續數值->綠燈秒數)
#   ])
#   # optimizer=Adam 是優化器，用來更新模型權重，讓損失函數（誤差）變小
#   # learning_rate=0.001 是更新的速度，太大可能不穩定，太小學得慢
#   # loss='mse' 是損失函數，用「均方誤差」衡量預測值和真實值差距
#   model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mse', metrics = ['mae'])
#   return model


# ========================
# ✅ 強化版本：支援 Dropout、L2 正則化、參數設定
# ========================
def build_model(input_shape, dropout_rate = 0.2, l2_factor = 0.001):
  model = Sequential([
      Input(shape = ( input_shape,)),
      Dense(64, activation = 'relu', kernel_regularizer = regularizers.l2(l2_factor)),
      Dropout(dropout_rate),
      Dense(32, activation = 'relu', kernel_regularizer = regularizers.l2(l2_factor)),
      Dropout(dropout_rate),
      Dense(1),
  ])
  model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mse', metrics = ['mae'])
  return model


# ========================
# 模型訓練
# ========================
def train_model(model, X_train, y_train, epochs = 50):
  early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, min_delta = 0.001, restore_best_weights = True)

  reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-6, verbose = 1)

  history = model.fit(X_train, y_train, epochs = epochs, batch_size = 32, validation_split = 0.2, callbacks = [ early_stop, reduce_lr ], verbose = 1)

  return history


# ========================
# 評估訓練與測試集
# ========================
def evaluate_train(model, X_train, y_train):
  y_pred = model.predict(X_train)
  mse = mean_squared_error(y_train, y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_train, y_pred)
  r2 = r2_score(y_train, y_pred)

  print("訓練集模型評估結果：")
  print(f"📉 MSE: {mse:.2f}")
  print(f"📉 RMSE: {rmse:.2f}")
  print(f"📉 MAE: {mae:.2f}")
  print(f"📈 R² : {r2:.4f}")

  return y_pred


def evaluate_test(model, X_test, y_test):
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  print("測試集模型評估結果：")
  print(f"📉 MSE: {mse:.2f}")
  print(f"📉 RMSE: {rmse:.2f}")
  print(f"📉 MAE: {mae:.2f}")
  print(f"📈 R² : {r2:.4f}")

  return y_pred


# ========================
# 預測新資料
# ========================
def predict_new(model, X_new, min_sec = 20, max_sec = 99, float_output = False):
  predictions = model.predict(X_new)
  if not float_output:
    predictions = np.round(predictions).astype(int)
  predictions = np.clip(predictions, min_sec, max_sec)
  return predictions
