# shap_utils.py
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def explain_shap_feature(model, X_train, X_test, feature_names, output_dir = "shap", nsamples = 100):
  try:
    print("📊 開始使用 SHAP KernelExplainer 計算特徵重要性...")
    os.makedirs(output_dir, exist_ok = True)

    # 用部分測試資料作為樣本
    X_sample = pd.DataFrame(X_test[: 100], columns = feature_names)

    # 使用訓練資料中隨機 100 筆作為背景資料
    background_size = X_train.shape[0] // 200  # 取訓練資料 1/ 200 作為背景資料大小

    background = X_train[np.random.choice(X_train.shape[0], background_size, replace = False)]

    # 定義模型預測函數
    def model_predict(data_as_numpy):
      return model.predict(data_as_numpy).flatten()

    # 初始化 SHAP 解釋器並計算 shap 值
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_sample, nsamples = nsamples)

    # SHAP 條形圖
    shap.summary_plot(shap_values, X_sample, plot_type = "bar", show = False)
    plt.savefig(f"{output_dir}/shap_feature_importance_bar.png")
    plt.clf()

    # SHAP 蜂群圖
    shap.summary_plot(shap_values, X_sample, show = False)
    plt.savefig(f"{output_dir}/shap_feature_importance_beeswarm.png")
    plt.clf()

    # 計算特徵重要性並列印前20名
    mean_abs_shap = np.mean(np.abs(shap_values), axis = 0)
    feature_importance_df = pd.DataFrame({ 'feature': feature_names, 'mean_abs_shap_value': mean_abs_shap}).sort_values(by = 'mean_abs_shap_value', ascending = False).reset_index(drop = True)

    print("\n📝 SHAP 特徵重要性（平均絕對 SHAP 值）排序（前20名）：")
    print(feature_importance_df.head(20))
    print(f"✅ SHAP 圖已儲存至 '{output_dir}' 資料夾")

    return feature_importance_df

  except Exception as e:
    print("❌ SHAP 解釋失敗，原因：", e)
    return None
