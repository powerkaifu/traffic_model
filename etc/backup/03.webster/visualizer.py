# visualizer.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 在圖表中顯示微軟正黑體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_scatter_predictions(y_true, y_pred):
    """
    繪製「預測值 vs 實際值」散點圖
    """
    plt.figure(figsize=(8,8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='理想預測線')
    plt.xlabel("實際值")
    plt.ylabel("預測值")
    plt.title("📊 預測值 vs 實際值")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# M:機車 S:小型車 L:大型車 T:聯結車
def plot_volume_distribution(df):
    plt.figure(figsize=(12, 8))
    sns.histplot(df['Volume_M'], color='orange', kde=True, bins=30, label='機車')
    sns.histplot(df['Volume_S'], color='blue', kde=True, bins=30, label='小型車')
    sns.histplot(df['Volume_L'], color='green', kde=True, bins=30, label='大型車')
    sns.histplot(df['Volume_T'], color='red', kde=True, bins=30, label='聯結車')
    plt.title("各類車型流量分布")
    plt.xlabel("流量")
    plt.legend()
    plt.show()

def plot_speed_distribution(df):
    plt.figure(figsize=(12, 8))
    sns.histplot(df['Speed_M'], color='orange', kde=True, bins=30, label='機車')
    sns.histplot(df['Speed_S'], color='blue', kde=True, bins=30, label='小型車')
    sns.histplot(df['Speed_L'], color='green', kde=True, bins=30, label='大型車')
    sns.histplot(df['Speed_T'], color='red', kde=True, bins=30, label='聯結車')
    plt.title("各類車型速度分布")
    plt.xlabel("速度")
    plt.legend()
    plt.show()