# predictor.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from visualizer import plot_scatter_predictions

'''
一個基於神經網路的監督式學習 (Supervised Learning) 模型
用於回歸任務 (Regression) 來預測連續變數
'''

# 設定字型為微軟正黑體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 確保負號能顯示正常


# 建立模型
def build_model(input_shape):
  # Sequential() 是 Keras 中的一種模型類型，表示一個線性堆疊的神經網路。
  model = Sequential([
      Input(shape = ( input_shape,)),
      Dense(64, activation = 'relu'),
      Dense(32, activation = 'relu'),
      Dense(1)  # 預測一個連續值：綠燈秒數
  ])
  # 將模型的損失函數設定為 'mse'，均方誤差是衡量回歸模型預測值與真實值之間差異的常用指標
  model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mse')
  return model


# 訓練模型
'''
model 要訓練的模型，通常是使用 Keras 建立的深度學習模型，例如 Sequential() 或是其他模型架構。
X_train 訓練資料的輸入特徵，通常是數組或張量（如 NumPy 陣列或 TensorFlow 張量）。這些是模型用來學習的資料（例如影像像素值、特徵向量等）。
y_train 訓練資料的目標（標籤）。對於分類問題，它是每個樣本對應的類別標籤；對於回歸問題，它是每個樣本的數值目標。
epochs 訓練過程中模型會遍歷訓練資料集的次數。每一次遍歷訓練資料集稱為一個 "epoch"。更多的 epoch 意味著模型會學習更多次，但也可能會導致過擬合。
batch_size 訓練過程中，資料會被分割成小批次來進行處理。batch_size 指的是每次進行梯度更新時所使用的訓練樣本數量。
verbose 這是用來設置訓練過程中輸出訊息的詳細程度。0: 不輸出訊息。1: 顯示進度條和訓練過程。2: 顯示每個 epoch 的簡單訊息。
'''


def train_model(model, X_train, y_train, epochs = 50):
  model.fit(X_train, y_train, epochs = epochs, batch_size = 32, verbose = 1)


# 評估模型
'''
MSE（均方誤差，Mean Squared Error）
越小表示模型的預測誤差越小。由於它對大誤差（離群點）特別敏感，當 MSE 值小時，模型能夠更準確地預測大多數樣本的結果。

RMSE（均方根誤差，Root Mean Squared Error）
越小表示模型的預測誤差越小，並且能夠與實際數據的範圍相比較。由於它的單位和數據相同，因此可以更直觀地理解模型的誤差大小。

MAE（平均絕對誤差，Mean Absolute Error）
越小表示模型的預測誤差越小，且對於較小誤差更加敏感。由於 MAE 不對誤差平方處理，因此對異常值（離群點）的影響較小。

R²（決定係數，R-squared）
越大表示模型解釋數據變異的能力越強，即模型越能捕捉到實際數據中的模式。數值越接近 1，說明模型的預測非常接近實際值。
如果 R² 的值接近 0，則意味著模型幾乎沒有解釋數據的變異，預測效果不好。負值的 R² 甚至表示模型的表現比簡單的平均數模型還差。
'''


def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)

  mse = mean_squared_error(y_test, y_pred)  # 均方誤差
  rmse = np.sqrt(mse)  # 均方根誤差
  mae = mean_absolute_error(y_test, y_pred)  # 平均絕對誤差
  r2 = r2_score(y_test, y_pred)  # 決定係數

  print("模型評估結果：")
  print(f"📉 MSE（均方誤差）: {mse:.2f}")
  print(f"📉 RMSE（均方根誤差）: {rmse:.2f}")
  print(f"📉 MAE（平均絕對誤差）: {mae:.2f}")
  print(f"📈 R²（決定係數）: {r2:.4f}")

  # 散點圖：預測值 vs 實際值
  plot_scatter_predictions(y_test.flatten(), y_pred.flatten())

  return y_pred


# 預測新資料
def predict_new(model, X_new):
  predictions = model.predict(X_new)
  return np.round(predictions)
