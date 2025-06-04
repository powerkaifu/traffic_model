def assign_green_seconds(df, n_phases = 2, saturation_flow = 1900, loss_time_per_phase = 4):
  # 保留原欄位名稱，方便最後還原欄位順序
  original_columns = df.columns.tolist()

  # 補空值處理，避免後續計算錯誤
  cols_to_fill_zero = [ 'Volume_S', 'Volume_L', 'Volume_T', 'Occupancy', 'Speed_S', 'Speed_L', 'Speed_T', 'IsPeakHour']
  for col in cols_to_fill_zero:
    if col in df.columns:
      df[col] = df[col].fillna(0)

  # 確保 IsPeakHour 是整數型態
  if 'IsPeakHour' in df.columns:
    df['IsPeakHour'] = df['IsPeakHour'].astype(int)

  # 特徵交互：新增乘積特徵，擴充資料表示能力
  if all(col in df.columns for col in [ 'Occupancy', 'Volume_S']):
    df['Occ_x_Volume_S'] = df['Occupancy'] * df['Volume_S']
  if all(col in df.columns for col in [ 'Occupancy', 'Volume_L']):
    df['Occ_x_Volume_L'] = df['Occupancy'] * df['Volume_L']
  if all(col in df.columns for col in [ 'Occupancy', 'Volume_T']):
    df['Occ_x_Volume_T'] = df['Occupancy'] * df['Volume_T']
  if all(col in df.columns for col in [ 'Speed_S', 'Volume_S']):
    df['SpeedS_x_VolumeS'] = df['Speed_S'] * df['Volume_S']
  if all(col in df.columns for col in [ 'Speed_L', 'Volume_L']):
    df['SpeedL_x_VolumeL'] = df['Speed_L'] * df['Volume_L']
  if all(col in df.columns for col in [ 'Speed_T', 'Volume_T']):
    df['SpeedT_x_VolumeT'] = df['Speed_T'] * df['Volume_T']

  # 核心：根據 Webster 理論計算加權流量和流量比率
  vehicle_weights_webster = { 'Volume_S': 1.0, 'Volume_L': 1.5, 'Volume_T': 2.0}
  df['_weighted_flow'] = df.apply(lambda row: sum(row[v] * w for v, w in vehicle_weights_webster.items() if v in row), axis = 1)
  df['_flow_ratio_yi'] = df['_weighted_flow'] / saturation_flow if saturation_flow > 0 else 0.0

  # 計算總流量比率與最大限制，避免過大造成週期計算錯誤
  total_flow_ratio_Y = df['_flow_ratio_yi'].sum()
  total_flow_ratio_Y = min(total_flow_ratio_Y, 0.95)

  # 計算損失時間，包含紅燈切換損失等
  total_loss_time_L = n_phases * loss_time_per_phase

  # 根據 Webster 公式計算號誌週期長度，防止分母過小
  denom = 1 - total_flow_ratio_Y
  if denom < 0.05:
    cycle_length_C = 120
  else:
    cycle_length_C = (1.5 * total_loss_time_L + 5) / denom
    cycle_length_C = max(30, min(cycle_length_C, 120))

  # 若流量為零，平均分配綠燈時間；否則依流量比率分配綠燈秒數
  if total_flow_ratio_Y == 0:
    df['_green_base'] = (cycle_length_C - total_loss_time_L) / max(n_phases, 1)
  else:
    df['_green_base'] = (cycle_length_C - total_loss_time_L) * (df['_flow_ratio_yi'] / total_flow_ratio_Y)
  df['_green_base'] = df['_green_base'].clip(lower = 0)

  # 基礎調整：根據占有率調整綠燈秒數
  base_adjustment = 20 + (df['Occupancy'] / 100) * 70 if 'Occupancy' in df.columns else 20

  # 車種加權獎勵，根據車輛數量給予額外加分
  vehicle_weights_bonus = { 'Volume_S': 2.0, 'Volume_L': 2.5, 'Volume_T': 3.0}
  vehicle_bonus = df.apply(lambda row: sum(row[v] * w for v, w in vehicle_weights_bonus.items() if v in row), axis = 1)

  # 速度懲罰，速度低於 30 的車輛每種加 5 秒綠燈時間
  def speed_penalty(row):
    return sum(5 for s in [ 'Speed_S', 'Speed_L', 'Speed_T'] if s in row and row[s] < 30)

  speed_penalty = df.apply(speed_penalty, axis = 1)

  # 尖峰時段加分，增加綠燈秒數
  peak_bonus = df['IsPeakHour'] * 20 if 'IsPeakHour' in df.columns else 0

  # 綜合所有影響因素，計算最終綠燈秒數，並限制上下界
  green_seconds = df['_green_base'] + base_adjustment + vehicle_bonus + speed_penalty + peak_bonus
  green_seconds = green_seconds.clip(lower = 20, upper = 99).round().astype(int)

  # 將結果寫回 DataFrame，新增欄位 'green_seconds'
  df['green_seconds'] = green_seconds

  # 清理中間計算用的暫存欄位
  for temp_col in [ '_weighted_flow', '_flow_ratio_yi', '_green_base']:
    if temp_col in df.columns:
      df.drop(columns = temp_col, inplace = True)

  # 確保欄位順序不變，並保留交互特徵與新欄位
  extra_cols = [ col for col in df.columns if col not in original_columns and col != 'green_seconds']
  df = df[original_columns + extra_cols + ['green_seconds']]

  return df
