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
    'plot_volume_distribution',
    'plot_speed_distribution',
    'plot_occupancy_distribution',
    'plot_occupancy_vs_green_seconds',
    'plot_occupancy_time_trend',
    'plot_residuals',
]


# 📌 散點圖（plot_scatter_predictions）
# 功能：直觀呈現「模型預測值」與「Webster 函數產生的實際值」之間的吻合程度。
# 點落在紅色虛線（y = x）上表示預測與實際完全相符。
# 點落在線附近表示模型成功模仿了函數行為。
def plot_scatter_predictions(y_true, y_pred):
  plt.figure(figsize = ( 8, 8 ))
  plt.scatter(y_true, y_pred, alpha = 0.5)
  mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
  plt.plot([ mn, mx ], [ mn, mx ], 'r--', label = '理想預測線')
  plt.xlabel("實際值")
  plt.ylabel("預測值")
  plt.title("📊 預測值 vs 實際值")
  plt.legend()
  plt.grid(True)
  plt.axis("equal")
  plt.tight_layout()
  plt.show()


# 📌 流量分布圖（plot_volume_distribution）- 特徵分佈類
# 功能：顯示不同車型的流量分布情況。
# ✅ 幫助理解各類車型在資料集中所占比例，確認訓練資料的合理性與均衡性。
# ✅ 為模型判斷交通狀況提供背景依據，也可用來說明資料分布是否對模型造成偏倚。
def plot_volume_distribution(df):
  plt.figure(figsize = ( 12, 8 ))
  sns.histplot(df['Volume_M'], color = 'orange', kde = True, bins = 30, label = '機車')
  sns.histplot(df['Volume_S'], color = 'blue', kde = True, bins = 30, label = '小型車')
  sns.histplot(df['Volume_L'], color = 'green', kde = True, bins = 30, label = '大型車')
  sns.histplot(df['Volume_T'], color = 'red', kde = True, bins = 30, label = '聯結車')
  plt.title("各類車型流量分布")
  plt.xlabel("流量")
  plt.legend()
  plt.show()


# 📌 速度分布圖（plot_speed_distribution）- 特徵分佈類
# 功能：顯示各類車型的速度分布情形。
# ✅ 分析車種速度差異對綠燈配時是否有合理區分。
# ✅ 作為資料探索與特徵影響力分析的基礎，並輔助後續模型特徵重要性說明。
def plot_speed_distribution(df):
  plt.figure(figsize = ( 12, 8 ))
  sns.histplot(df['Speed_M'], color = 'orange', kde = True, bins = 30, label = '機車')
  sns.histplot(df['Speed_S'], color = 'blue', kde = True, bins = 30, label = '小型車')
  sns.histplot(df['Speed_L'], color = 'green', kde = True, bins = 30, label = '大型車')
  sns.histplot(df['Speed_T'], color = 'red', kde = True, bins = 30, label = '聯結車')
  plt.title("各類車型速度分布")
  plt.xlabel("速度")
  plt.legend()
  plt.show()


# 📌 佔有率分布圖（plot_occupancy_distribution）- 特徵分佈類
# 功能：顯示佔有率（Occupancy）的分布情況。
# ✅ 幫助理解交通流量的佔用情況，確認資料集是否有合理的佔有率分布。
def plot_occupancy_distribution(df):
  plt.figure(figsize = ( 10, 6 ))
  sns.histplot(df['Occupancy'], bins = 30, kde = True, color = 'teal')
  plt.title("佔有率（Occupancy）分布圖")
  plt.xlabel("Occupancy (%)")
  plt.ylabel("頻率")
  plt.grid(True)
  plt.show()


# 📌 佔有率與綠燈秒數散點圖（plot_occupancy_vs_green_seconds
# 功能：顯示佔有率與綠燈秒數之間的關係。
# ✅ 幫助理解佔有率對綠燈配時的影響，確認模型是否合理考慮了佔有率因素。
def plot_occupancy_vs_green_seconds(df):
  plt.figure(figsize = ( 10, 6 ))
  plt.scatter(df['Occupancy'], df['green_seconds'], alpha = 0.5, color = 'purple')
  plt.title("佔有率 vs 綠燈秒數散點圖")
  plt.xlabel("Occupancy (%)")
  plt.ylabel("綠燈秒數 (seconds)")
  plt.grid(True)
  plt.show()


# 📌 佔有率隨時間變化趨勢圖（plot_occupancy_time_trend）
# 功能：顯示佔有率隨時間變化的趨勢。
# ✅ 幫助分析佔有率在不同時間段的變化情況，確認是否存在高峰時段。
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


# 📌 散點圖（plot_scatter_predictions）
# 功能：直觀呈現「模型預測值」與「Webster 函數產生的實際值」之間的吻合程度。
# 分析重點：
# - 點落在紅色虛線（y = x）上表示預測與實際完全相符。
# - 點落在線附近表示模型成功模仿了函數行為，但不是直接套用。
# 使用意義：
# ✅ 可用於向指導老師說明：模型是透過學習預測趨勢，而非硬套公式。
# ✅ 本圖為教師模仿（imitation learning）的有效視覺化證明。
