import pandas as pd
import numpy as np


def assign_green_seconds(
    df: pd.DataFrame,
    n_phases: int = 2,  # 路口時相數量 (這裡仍為參考值，因模型輸出為單一秒數)
    saturation_flow: int = 1900,  # 飽和流量 (veh/h)
    loss_time_per_phase: int = 4,  # 每時相損失時間 (秒)
    min_green_seconds: int = 20,  # 最小綠燈秒數
    max_green_seconds: int = 99,  # 最大綠燈秒數
    min_cycle_length: int = 30,  # 最小週期長度
    max_cycle_length: int = 120,  # 最大週期長度
    # 動態調整策略參數 (可根據實驗調整)
    occupancy_base_seconds: float = 20.0,  # 佔有率調整的基礎秒數
    occupancy_max_bonus: float = 70.0,  # 佔有率能提供的最大額外秒數
    volume_bonus_per_unit: float = 0.005,  # 每單位加權車輛數的綠燈獎勵秒數
    speed_penalty_threshold: float = 30.0,  # 低於此速度則施加懲罰 (km/h)
    speed_penalty_amount: float = 5.0,  # 速度懲罰的固定額外秒數
    peak_hour_bonus_seconds: int = 20  # 尖峰時段額外加成秒數
) -> pd.DataFrame:
  """
    結合 Webster 號誌配時理論與實時交通數據(特徵值)，計算並分配綠燈秒數。
    此函數將利用提供的整體路口特徵，推算一個綜合的綠燈秒數作為弱標籤。

    參數:
    df (pd.DataFrame): 包含交通數據的 DataFrame，應包含以下欄位：
                       'Volume_S', 'Volume_L', 'Volume_T',
                       'Occupancy', 'Speed_S', 'Speed_L', 'Speed_T',
                       'IsPeakHour' (布林值或 0/1)。
    n_phases (int): 路口時相數量 (通常為2，這裡主要用於週期長度計算)。
    saturation_flow (int): 飽和流量 (veh/h)。
    loss_time_per_phase (int): 每時相損失時間 (秒)。
    min_green_seconds (int): 最小綠燈秒數限制。
    max_green_seconds (int): 最大綠燈秒數限制。
    min_cycle_length (int): 最小週期長度限制。
    max_cycle_length (int): 最大週期長度限制。
    occupancy_base_seconds (float): 佔有率調整的基礎秒數。
    occupancy_max_bonus (float): 佔有率能提供的最大額外秒數。
    volume_bonus_per_unit (float): 每單位加權車輛數的綠燈獎勵秒數。
    speed_penalty_threshold (float): 低於此速度則施加懲罰 (km/h)。
    speed_penalty_amount (float): 速度懲罰的固定額外秒數。
    peak_hour_bonus_seconds (int): 尖峰時段額外加成秒數。

    回傳:
    pd.DataFrame: 含新增 'green_seconds' 弱標籤欄位。
    """

  # --- 預處理與數據補零 ---
  # 確保所有預期會用到的欄位存在，不存在則填充為 0
  expected_cols = [ 'Volume_S', 'Volume_L', 'Volume_T', 'Occupancy', 'Speed_S', 'Speed_L', 'Speed_T', 'IsPeakHour']
  for col in expected_cols:
    if col not in df.columns:
      df[col] = 0.0  # 使用浮點數以保持一致性

  # 確保 IsPeakHour 是整數類型
  if 'IsPeakHour' in df.columns:
    df['IsPeakHour'] = df['IsPeakHour'].astype(int)

  # --- 加權流量計算 (Webster) ---
  # 這些權重仍可用於計算整體加權流量
  vehicle_weights_webster = { 'Volume_S': 1.0, 'Volume_L': 1.5, 'Volume_T': 2.0}

  # 使用 apply 函數計算每行的加權流量
  # np.sum 是為了處理 df[v_col] 可能不存在或為空 Series 的情況
  df['weighted_flow'] = df.apply(lambda row: np.sum([row[v_col] * weight if v_col in row else 0 for v_col, weight in vehicle_weights_webster.items()]), axis = 1)

  # 流量比 y_i
  # 避免 saturation_flow 為零導致錯誤
  df['flow_ratio_yi'] = np.where(saturation_flow <= 0, 0.0, df['weighted_flow'] / saturation_flow)

  # 總流量比 Y (對於單一輸出，這裡是所有流量比的總和，但需按行計算，不是整個 DataFrame 的 sum())
  # 這裡假設每一行 df 都是一個獨立的觀察，且其 weighted_flow 代表該觀察的總流量
  # Webster 理論的 Y 是各時相流量比的總和，如果 df 每行是單一時相的流量，則多行求和才是 Y
  # 但如果 df 每行是路口整體流量，則該行 Y 就直接是其 flow_ratio_yi (這與您原始代碼邏輯一致)
  # 由於您要求 df 特徵值不變，這裡維持原始邏輯，將 df['flow_ratio_yi'] 作為單一觀察點的 Y

  # 為了計算 Cycle Length，我們需要一個代表 "總流量比" 的 Y。
  # 如果 df 的每一行代表一個路口在一個時間點的整體觀察，
  # 那麼 df['flow_ratio_yi'] 本身就代表了該觀察點的「主要」流量壓力。
  # 原始代碼中 `total_flow_ratio_Y = df['flow_ratio_yi'].sum()` 這行
  # 暗示了將所有觀察點的流量比加起來，這在為單個觀察點生成秒數時可能不符 Webster 本意。
  # 這裡我們修正為針對每行計算其 `total_flow_ratio_Y`，即為該行自己的 `flow_ratio_yi`。

  # 修正後的總流量比 Y
  # 每個時間點的總流量比 Y 就是該時間點的流量比 y_i
  total_flow_ratio_Y_per_row = df['flow_ratio_yi'].clip(upper = 0.95)

  # 總損失時間 L
  total_loss_time_L = n_phases * loss_time_per_phase

  # 計算週期長度 C (Cycle Length)
  denom = 1 - total_flow_ratio_Y_per_row
  cycle_length_C = np.where(
      denom < 0.05,  # 分母接近0，表示流量接近飽和
      max_cycle_length,  # 取上限週期
      (1.5 * total_loss_time_L + 5) / denom
  )
  cycle_length_C = np.clip(cycle_length_C, min_cycle_length, max_cycle_length)  # 限制週期長度範圍

  # 基礎綠燈秒數 g_i
  # 由於我們沒有明確區分 EW/NS，這裡假設綠燈時間是在兩個主要時相間平均分配
  # 或者說，這是「每個主要時相」的基礎綠燈時間，然後再進行調整
  effective_green_webster = np.where(
      total_flow_ratio_Y_per_row == 0,
      (cycle_length_C - total_loss_time_L) / max(n_phases, 1),  # 平均分配
      (cycle_length_C - total_loss_time_L) * (df['flow_ratio_yi'] / total_flow_ratio_Y_per_row)
  )
  effective_green_webster = np.clip(effective_green_webster, 0, None)

  # --- 動態調整 ---
  final_green_seconds = pd.Series(effective_green_webster, index = df.index)

  # 1. 佔有率調整
  if 'Occupancy' in df.columns:
    # 從 occupancy_base_seconds 開始，根據佔有率增加秒數
    # 0% 佔有率給 base_seconds，100% 佔有率給 base_seconds + max_bonus
    adjustment_occupancy = occupancy_base_seconds + (df['Occupancy'] / 100) * occupancy_max_bonus
    final_green_seconds += adjustment_occupancy
  else:
    final_green_seconds += occupancy_base_seconds  # 如果沒有佔有率，只加基礎值

  # 2. 車輛數量加成 (使用原始權重，並乘以可配置的獎勵因子)
  total_volume_bonus = df.apply(lambda row: np.sum([row[v_col] * weight if v_col in row else 0 for v_col, weight in vehicle_weights_webster.items()]), axis = 1) * volume_bonus_per_unit
  final_green_seconds += total_volume_bonus

  # 3. 速度懲罰 (每個方向低於 speed_penalty_threshold 時，各加 speed_penalty_amount 秒懲罰)
  # 這裡我們處理為只要有任何一個方向速度低於閾值，就施加懲罰
  # (因為原始 df 沒有區分 EW/NS，只能這樣做)
  speed_penalty_active = False
  for s_col in [ 'Speed_S', 'Speed_L', 'Speed_T']:
    if s_col in df.columns and (df[s_col] < speed_penalty_threshold).any():
      speed_penalty_active = True
      break  # 只要有一個方向觸發懲罰，就施加

  if speed_penalty_active:
    # 對於那些至少有一個速度低於閾值的行，施加懲罰
    # 這裡的邏輯是，如果某行有任一速度低於閾值，就給予該行的綠燈時間一個固定的懲罰
    # 更精細的方式可能是累加每個低速車流的懲罰，但受限於原始 df 特徵
    df['speed_low_flag'] = 0
    for s_col in [ 'Speed_S', 'Speed_L', 'Speed_T']:
      if s_col in df.columns:
        df['speed_low_flag'] = np.where(df[s_col] < speed_penalty_threshold, 1, df['speed_low_flag'])

    final_green_seconds += df['speed_low_flag'] * speed_penalty_amount
    df = df.drop(columns = ['speed_low_flag'], errors = 'ignore')

  # 4. 尖峰時段加成
  if 'IsPeakHour' in df.columns:
    final_green_seconds += df['IsPeakHour'] * peak_hour_bonus_seconds

  # --- 最終綠燈秒數 ---
  # 限制綠燈秒數範圍並四捨五入
  df['green_seconds'] = final_green_seconds.clip(lower = min_green_seconds, upper = max_green_seconds).round().astype(int)

  # 清理暫存欄位
  cols_to_drop = [ 'weighted_flow', 'flow_ratio_yi', 'effective_green_webster']
  df = df.drop(columns = cols_to_drop, errors = 'ignore')

  return df


# --- 範例使用方式 (使用原始的特徵命名) ---
if __name__ == "__main__":
  # 創建一個模擬的交通數據 DataFrame，使用原始的特徵命名
  data = {
      'Volume_S': [ 100, 200, 50, 300, 10, 150 ],
      'Volume_L': [ 30, 80, 20, 120, 5, 40 ],
      'Volume_T': [ 10, 40, 10, 60, 2, 15 ],
      'Occupancy': [ 20, 60, 15, 80, 5, 45 ],
      'Speed_S': [ 60, 20, 50, 15, 70, 35 ],
      'Speed_L': [ 40, 10, 30, 5, 60, 25 ],
      'Speed_T': [ 50, 30, 45, 10, 65, 30 ],
      'IsPeakHour': [ 0, 1, 0, 1, 0, 1 ]  # 增加一個數據點
  }
  df_traffic = pd.DataFrame(data)

  print("原始數據:")
  print(df_traffic)
  print("\n" + "=" * 50 + "\n")

  # 調用函數生成弱標籤
  # 您可以調整這些參數來觀察對生成秒數的影響
  df_with_weak_label = assign_green_seconds(
      df_traffic.copy(),  # 傳入副本以避免修改原始df
      saturation_flow = 1800,  # 可調整
      loss_time_per_phase = 5,  # 可調整
      occupancy_base_seconds = 25.0,  # 調整佔有率調整的基礎秒數
      occupancy_max_bonus = 60.0,  # 調整佔有率能提供的最大額外秒數
      volume_bonus_per_unit = 0.007,  # 調整流量獎勵
      speed_penalty_amount = 7.0,  # 調整速度懲罰的固定額外秒數
      peak_hour_bonus_seconds = 25  # 調整尖峰時段加成
  )

  print("生成弱標籤後的數據 (只顯示部分特徵和生成的綠燈秒數):")
  # 為了清晰顯示，只選擇部分關鍵列
  output_cols = [ 'Volume_S', 'Occupancy', 'Speed_S', 'IsPeakHour', 'green_seconds']
  print(df_with_weak_label[output_cols])

  print("\n--- 觀察不同情況下的綠燈秒數 ---")
  print("案例 1 (正常流量，非尖峰):")
  print(df_with_weak_label.iloc[0][output_cols])

  print("\n案例 2 (高流量，低速度，尖峰時段):")
  print(df_with_weak_label.iloc[1][output_cols])

  print("\n案例 3 (低流量，正常速度，非尖峰):")
  print(df_with_weak_label.iloc[2][output_cols])

  print("\n案例 4 (極高流量，極低速度，尖峰時段):")
  print(df_with_weak_label.iloc[3][output_cols])

  print("\n案例 5 (非常低流量，高速度，非尖峰):")
  print(df_with_weak_label.iloc[4][output_cols])

  print("\n案例 6 (中等流量，部分低速，尖峰時段):")
  print(df_with_weak_label.iloc[5][output_cols])
