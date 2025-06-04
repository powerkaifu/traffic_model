def assign_green_seconds(df, n_phases = 2, saturation_flow = 1900, loss_time_per_phase = 4):
  """
    使用 Webster 號誌配時理論結合實時交通特徵，計算最適綠燈秒數（不修改 df 結構）
    - 核心：根據流量比 y_i 計算最適週期長度 C，再分配每個方向綠燈時間。
    - 加入：佔有率、速度、車種加權、尖峰時段 等因子作動態調整。
    """

  # 保留原欄位名稱
  original_columns = df.columns.tolist()

  # 補空值處理
  cols_to_fill_zero = [ 'Volume_S', 'Volume_L', 'Volume_T', 'Occupancy', 'Speed_S', 'Speed_L', 'Speed_T', 'IsPeakHour']
  for col in cols_to_fill_zero:
    if col in df.columns:
      df[col] = df[col].fillna(0)

  if 'IsPeakHour' in df.columns:
    df['IsPeakHour'] = df['IsPeakHour'].astype(int)

  # --- 核心：Webster 流量加權計算 ---
  vehicle_weights_webster = { 'Volume_S': 1.0, 'Volume_L': 1.5, 'Volume_T': 2.0}

  df['_weighted_flow'] = df.apply(lambda row: sum(row[v] * w for v, w in vehicle_weights_webster.items() if v in row), axis = 1)

  df['_flow_ratio_yi'] = df['_weighted_flow'] / saturation_flow if saturation_flow > 0 else 0.0
  total_flow_ratio_Y = df['_flow_ratio_yi'].sum()
  total_flow_ratio_Y = min(total_flow_ratio_Y, 0.95)
  total_loss_time_L = n_phases * loss_time_per_phase

  denom = 1 - total_flow_ratio_Y
  if denom < 0.05:
    cycle_length_C = 120
  else:
    cycle_length_C = (1.5 * total_loss_time_L + 5) / denom
    cycle_length_C = max(30, min(cycle_length_C, 120))

  if total_flow_ratio_Y == 0:
    df['_green_base'] = (cycle_length_C - total_loss_time_L) / max(n_phases, 1)
  else:
    df['_green_base'] = (cycle_length_C - total_loss_time_L) * (df['_flow_ratio_yi'] / total_flow_ratio_Y)
  df['_green_base'] = df['_green_base'].clip(lower = 0)

  # --- 動態調整 ---
  base_adjustment = 20 + (df['Occupancy'] / 100) * 70 if 'Occupancy' in df.columns else 20

  vehicle_weights_bonus = { 'Volume_S': 2.0, 'Volume_L': 2.5, 'Volume_T': 3.0}
  vehicle_bonus = df.apply(lambda row: sum(row[v] * w for v, w in vehicle_weights_bonus.items() if v in row), axis = 1)

  def speed_penalty(row):
    return sum(5 for s in [ 'Speed_S', 'Speed_L', 'Speed_T'] if s in row and row[s] < 30)

  speed_penalty = df.apply(speed_penalty, axis = 1)

  peak_bonus = df['IsPeakHour'] * 20 if 'IsPeakHour' in df.columns else 0

  # 綜合所有項目得出最終綠燈秒數
  green_seconds = df['_green_base'] + base_adjustment + vehicle_bonus + speed_penalty + peak_bonus
  green_seconds = green_seconds.clip(lower = 20, upper = 99).round().astype(int)

  # 輸出結果：建立綠燈欄位，保證不改變原 df 結構（不留痕跡）
  df['green_seconds'] = green_seconds

  # 清除中途欄位
  for temp_col in [ '_weighted_flow', '_flow_ratio_yi', '_green_base']:
    if temp_col in df.columns:
      df.drop(columns = temp_col, inplace = True)

  # 確保欄位順序沒變
  df = df[original_columns + ['green_seconds']]

  return df
