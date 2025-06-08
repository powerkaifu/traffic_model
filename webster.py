def assign_green_seconds(df, n_phases = 2, saturation_flow = 1900, loss_time_per_phase = 4):
  original_columns = df.columns.tolist()

  # 補空值處理
  cols_to_fill_zero = [ 'Volume_S', 'Volume_L', 'Volume_T', 'Occupancy', 'Speed_S', 'Speed_L', 'Speed_T', 'IsPeakHour']
  for col in cols_to_fill_zero:
    if col in df.columns:
      df[col] = df[col].fillna(0)

  if 'IsPeakHour' in df.columns:
    df['IsPeakHour'] = df['IsPeakHour'].astype(int)

  # 特徵交互
  df['Occ_x_Volume_S'] = df['Occupancy'] * df.get('Volume_S', 0)
  df['Occ_x_Volume_L'] = df['Occupancy'] * df.get('Volume_L', 0)
  df['Occ_x_Volume_T'] = df['Occupancy'] * df.get('Volume_T', 0)
  df['SpeedS_x_VolumeS'] = df.get('Speed_S', 0) * df.get('Volume_S', 0)
  df['SpeedL_x_VolumeL'] = df.get('Speed_L', 0) * df.get('Volume_L', 0)
  df['SpeedT_x_VolumeT'] = df.get('Speed_T', 0) * df.get('Volume_T', 0)

  # 加權流量與流量比率 (Webster)
  vehicle_weights_webster = { 'Volume_S': 1.0, 'Volume_L': 1.5, 'Volume_T': 2.0}
  df['_weighted_flow'] = df.apply(lambda row: sum(row[v] * w for v, w in vehicle_weights_webster.items() if v in row), axis = 1)
  df['_flow_ratio_yi'] = df['_weighted_flow'] / saturation_flow if saturation_flow > 0 else 0.0

  total_flow_ratio_Y = df['_flow_ratio_yi'].sum()
  total_flow_ratio_Y = min(total_flow_ratio_Y, 0.95)
  total_loss_time_L = n_phases * loss_time_per_phase

  # Webster cycle length
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

  # 基礎調整
  base_adjustment = 20 + (df['Occupancy'] / 100) * 70 if 'Occupancy' in df.columns else 20

  # 車種加權獎勵（限制上限）
  vehicle_weights_bonus = { 'Volume_S': 2.0, 'Volume_L': 2.5, 'Volume_T': 3.0}
  vehicle_bonus = df.apply(lambda row: sum(row[v] * w for v, w in vehicle_weights_bonus.items() if v in row), axis = 1)
  vehicle_bonus = vehicle_bonus.clip(lower = 0, upper = 50)

  # 速度懲罰（限制上限）
  def speed_penalty_func(row):
    return sum(5 for s in [ 'Speed_S', 'Speed_L', 'Speed_T'] if s in row and row[s] < 30)

  speed_penalty = df.apply(speed_penalty_func, axis = 1)
  speed_penalty = speed_penalty.clip(lower = 0, upper = 15)

  # 尖峰時段加分
  peak_bonus = df['IsPeakHour'] * 20 if 'IsPeakHour' in df.columns else 0

  # 嚴重壅塞補償（速度近 0 且占有率 > 80）
  df['is_jam'] = ((df[[ 'Speed_S', 'Speed_L', 'Speed_T']].min(axis = 1) < 5) & (df['Occupancy'] > 80)).astype(int)
  jam_bonus = df['is_jam'] * 20

  # 合併
  green_seconds = df['_green_base'] + base_adjustment + vehicle_bonus + speed_penalty + peak_bonus + jam_bonus

  # 若總 flow ratio 太低（交通空蕩），限制最高不超過 60
  if total_flow_ratio_Y < 0.1:
    green_seconds = green_seconds.clip(upper = 60)

  # 最終裁剪上下界
  green_seconds = green_seconds.clip(lower = 20, upper = 90).round().astype(int)

  df['green_seconds'] = green_seconds

  # 清除暫存欄位
  for temp_col in [ '_weighted_flow', '_flow_ratio_yi', '_green_base', 'is_jam']:
    if temp_col in df.columns:
      df.drop(columns = temp_col, inplace = True)

  extra_cols = [ col for col in df.columns if col not in original_columns and col != 'green_seconds']
  df = df[original_columns + extra_cols + ['green_seconds']]
  return df
