# predictor.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from visualizer import plot_scatter_predictions

# 建立模型
'''
建立一個神經網路回歸模型，輸入給的特徵數，經過兩層有非線性變換的隱藏層，
最後輸出一個連續值（綠燈秒數），用 MSE 來衡量預測誤差，並用 Adam 優化器來訓練模型。


神經網路通常分成三種層：
輸入層：接收原始資料（你的特徵值）
隱藏層：位於輸入層和輸出層之間，用來「處理」資料的層，隱藏層由很多神經元（節點）組成，每個神經元會接收輸入，做計算，並把結果傳給下一層。
輸出層：產生結果（你的綠燈秒數預測）

為什麼要「非線性變換」？
如果神經元只是做線性計算（像是加權總和），不管疊幾層，整個網路的輸出還是線性的，等同於只有一層線性模型，學不到複雜的關係。
所以必須加「非線性函數」（activation function），讓模型可以學到複雜、彎曲的資料模式。

ReLU (Rectified Linear Unit) 是一種常見的非線性激活函數，規則是：
輸入 > 0 時輸出原值
輸入 ≤ 0 時輸出 0
這個簡單的函數讓模型可以學習非線性的資料特性。
'''


# input_shape 是模型的輸入形狀，通常是特徵數量，特徵值是 16。
def build_model(input_shape):
  # Sequential() 是 Keras 中的一種模型類型，表示一個線性堆疊的神經網路。
  model = Sequential([
      # 輸入層，告訴模型輸入的特徵數量
      Input(shape = ( input_shape,)),
      # 第一層隱藏層，有 64 個神經元（節點），每個神經元會執行一個簡單的計算，並用 ReLU 激活函數來增加非線性（讓模型能學複雜模式）
      Dense(64, activation = 'relu'),
      # 第二層隱藏層，有 32 個神經元，功能同上
      Dense(32, activation = 'relu'),
      # 預測一個連續值：綠燈秒數
      Dense(1)
  ])
  # optimizer=Adam 是優化器，用來更新模型權重，讓損失函數（誤差）變小
  # learning_rate=0.001 是更新的速度，太大可能不穩定，太小學得慢
  # loss='mse' 是損失函數，用「均方誤差」衡量預測值和真實值差距
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
  # model.fit(X_train, y_train, epochs = epochs, batch_size = 32, verbose = 1)
  early_stop = EarlyStopping(
      monitor = 'val_loss',  # 驗證集的損失值（loss on validation set），它代表模型在沒看過的資料上的預測誤差。
      patience = 5,  # 如果 val_loss 5 輪沒進步就停止
      restore_best_weights = True  # 回復到最佳模型參數
  )

  model.fit(
      X_train,
      y_train,
      epochs = epochs,
      batch_size = 32,
      validation_split = 0.2,  # 拿出 20% 做驗證集，幫助模型在訓練期間即時監控是否過擬合
      callbacks = [early_stop],
      verbose = 1
  )


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
