# visualizer.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 在圖表中顯示微軟正黑體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

__all__ = [
    'plot_scatter_predictions',
    'plot_feature_distributions',
    'plot_hourly_distributions',
    'plot_occupancy_time_trend',
    'plot_residuals',
]


# 📌 散點圖預測結果（plot_scatter_predictions）- 模型評估類
## 評估訓練、測試集的預測結果
def plot_scatter_predictions(y_true, y_pred, ax = None, title = "散點圖"):
  if ax is None:
    ax = plt.gca()
  ax.scatter(y_true, y_pred, alpha = 0.5)
  lims = [np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()])]
  ax.plot(lims, lims, 'r--')  # y=x 參考線
  ax.set_xlabel("真實值")
  ax.set_ylabel("預測值")
  ax.set_title(title)
  ax.grid(True)


# 直方圖+KDE 曲線
def plot_feature_distributions(df, features):
  n = len(features)
  fig, axs = plt.subplots(1, n, figsize = (6 * n, 4))
  if n == 1:
    axs = [axs]  # 保證是 list 以便迴圈處理

  for ax, feature in zip(axs, features):
    if feature == "Occupancy":
      sns.histplot(df['Occupancy'], bins = 30, kde = True, color = 'teal', ax = ax)
      ax.set_title("佔有率（Occupancy）分布圖", fontsize = 14)
      ax.set_xlabel("Occupancy (%)", fontsize = 12)
      ax.set_ylabel("次數", fontsize = 12)

    elif feature == "Speed":
      sns.histplot(df['Speed_M'], color = 'orange', kde = True, bins = 30, label = '機車', ax = ax)
      sns.histplot(df['Speed_S'], color = 'blue', kde = True, bins = 30, label = '小型車', ax = ax)
      sns.histplot(df['Speed_L'], color = 'green', kde = True, bins = 30, label = '大型車', ax = ax)
      sns.histplot(df['Speed_T'], color = 'red', kde = True, bins = 30, label = '聯結車', ax = ax)
      ax.set_title("各類車型速度分布")
      ax.set_xlabel("速度")
      ax.set_ylabel("次數")
      ax.legend()

    elif feature == "Volume":
      sns.histplot(df['Volume_M'], color = 'orange', kde = True, bins = 30, label = '機車', ax = ax)
      sns.histplot(df['Volume_S'], color = 'blue', kde = True, bins = 30, label = '小型車', ax = ax)
      sns.histplot(df['Volume_L'], color = 'green', kde = True, bins = 30, label = '大型車', ax = ax)
      sns.histplot(df['Volume_T'], color = 'red', kde = True, bins = 30, label = '聯結車', ax = ax)
      ax.set_title("各類車型流量分布")
      ax.set_xlabel("流量")
      ax.set_ylabel("次數")
      ax.legend()

    ax.grid(True)


# 📌 箱型圖-每小時特徵分布圖（plot_hourly_distributions）
def plot_hourly_distributions(df, features):

  n = len(features)
  fig, axs = plt.subplots(1, n, figsize = (6 * n, 4))
  if n == 1:
    axs = [axs]

  for ax, feature in zip(axs, features):
    sns.boxplot(data = df, x = "hour", y = feature, ax = ax)
    ax.set_title(f"{feature} 每小時分布")
    ax.set_xlabel("小時 (0-23)")
    ax.set_ylabel(feature)
    ax.grid(True)


# 📌 佔有率隨時間變化趨勢圖（plot_occupancy_time_trend）
def plot_occupancy_time_trend(df):
  # 假設你有日期時間欄位或用 Hour 組合時間
  df['Time'] = df['Hour'] + df['Minute'] / 60
  plt.figure(figsize = ( 12, 6 ))
  sns.lineplot(x = 'Time', y = 'Occupancy', data = df, marker = 'o', color = 'navy')
  plt.title("佔有率隨時間變化趨勢")
  plt.xlabel("時間 (小時)")
  plt.ylabel("Occupancy (%)")
  plt.grid(True)
  plt.show()


# 📌 誤差分布圖（plot_residuals）
# 功能：視覺化「實際值 - 預測值」的分布（即殘差圖）。
# 理想情況下，誤差應接近常態分布，左右對稱，無明顯偏移。
# 若分布偏左/右或呈雙峰，可能代表模型尚有調整空間。
# 模型預測結果具有合理的泛化能力。
def plot_residuals(y_true, y_pred):
  residuals = y_true - y_pred
  plt.figure(figsize = ( 10, 6 ))
  sns.histplot(residuals, kde = True, color = "purple")
  plt.title("預測誤差分佈（實際值 - 預測值）")
  plt.xlabel("誤差")
  plt.ylabel("頻率")
  plt.grid(True)
  plt.tight_layout()
  plt.show()
