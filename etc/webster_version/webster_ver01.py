# webster.py
# 綠燈秒數策略 - Webster 號誌配時理論
# Webster 號誌配時理論 是一種經典的交通號誌配時公式，用來計算十字路口每個時相（如東西向、南北向）的最適綠燈秒數，
# 目標是：可以減少車輛延誤時間，提高路口通行效率
# 其核心公式是根據「飽和流量（saturation flow）」與「實際流量」來推算整體週期時間（Cycle Time），再分配每個方向該得的綠燈時間。

def assign_green_seconds(df, n_phases=2, saturation_flow=1900, loss_time_per_phase=4):
    """
    結合 Webster 號誌配時理論與實時交通數據(特徵值)，計算並分配綠燈秒數。
    參數:
    df (pd.DataFrame): 包含交通數據的 DataFrame，應包含以下欄位：
                       'Volume_S', 'Volume_L', 'Volume_T',
                       'Occupancy', 'Speed_S', 'Speed_L', 'Speed_T',
                       'IsPeakHour' (布林值或 0/1)。
    n_phases (int): 路口時相數量 (通常為2)，指東西或南北方向。
    saturation_flow (int): 飽和流量 (veh/h)。
    loss_time_per_phase (int): 每時相損失時間 (秒)。

    回傳:
    pd.DataFrame: 含新增 'green_seconds' 欄位。
    """

    # --- 預處理 ---
    cols_to_fill_zero = ['Volume_S', 'Volume_L', 'Volume_T', 'Occupancy',
                         'Speed_S', 'Speed_L', 'Speed_T', 'IsPeakHour']
    for col in cols_to_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if 'IsPeakHour' in df.columns:
        df['IsPeakHour'] = df['IsPeakHour'].astype(int)

    # --- 加權流量計算 (Webster) ---
    vehicle_weights_webster = {'Volume_S': 1.0, 'Volume_L': 1.5, 'Volume_T': 2.0}

    def calc_weighted_flow(row):
        total = 0
        for v_col, weight in vehicle_weights_webster.items():
            val = row[v_col] if v_col in row else 0
            total += val * weight
        return total

    df['weighted_flow'] = df.apply(calc_weighted_flow, axis=1)

    # 流量比 y_i
    if saturation_flow <= 0:
        df['flow_ratio_yi'] = 0.0
    else:
        df['flow_ratio_yi'] = df['weighted_flow'] / saturation_flow

    # 總流量比 Y，限制最大為0.95
    total_flow_ratio_Y = df['flow_ratio_yi'].sum()
    total_flow_ratio_Y = min(total_flow_ratio_Y, 0.95)

    # 總損失時間 L
    total_loss_time_L = n_phases * loss_time_per_phase

    # 計算週期長度 C
    denom = 1 - total_flow_ratio_Y
    if denom < 0.05:
        cycle_length_C = 120  # 上限週期秒數
    else:
        cycle_length_C = (1.5 * total_loss_time_L + 5) / denom
        cycle_length_C = max(30, min(cycle_length_C, 120))  # 限制在30~120秒

    # 基礎綠燈秒數 g_i
    if total_flow_ratio_Y == 0:
        df['effective_green_webster'] = (cycle_length_C - total_loss_time_L) / max(n_phases, 1)
    else:
        df['effective_green_webster'] = (cycle_length_C - total_loss_time_L) * (df['flow_ratio_yi'] / total_flow_ratio_Y)
    df['effective_green_webster'] = df['effective_green_webster'].clip(lower=0)

    # --- 動態調整 ---
    # 1. 基本綠燈秒數（依佔有率調整）
    if 'Occupancy' in df.columns:
        base_adjustment = 20 + (df['Occupancy'] / 100) * 70  # 20~90秒區間
    else:
        base_adjustment = 20

    # 2. 車輛數量加成
    vehicle_weights_bonus = {'Volume_S': 2.0, 'Volume_L': 2.5, 'Volume_T': 3.0}
    def calc_vehicle_bonus(row):
        total = 0
        for v_col, weight in vehicle_weights_bonus.items():
            val = row[v_col] if v_col in row else 0
            total += val * weight
        return total
    vehicle_bonus = df.apply(calc_vehicle_bonus, axis=1)

    # 3. 速度懲罰 (每個方向低於 30km/h 時，各加 5 秒懲罰)
    def calc_speed_penalty(row):
        penalty = 0
        for s_col in ['Speed_S', 'Speed_L', 'Speed_T']:
            if s_col in row and row[s_col] < 30:
                penalty += 5
        return penalty

    # 速度懲罰計算
    speed_penalty = df.apply(calc_speed_penalty, axis=1)

    # 4. 尖峰時段加成
    peak_bonus = df['IsPeakHour'] * 20 if 'IsPeakHour' in df.columns else 0

    # --- 最終綠燈秒數 ---
    final_green_seconds = df['effective_green_webster'] + base_adjustment + vehicle_bonus + speed_penalty + peak_bonus

    # 限制綠燈秒數範圍 20~99 秒並四捨五入
    df['green_seconds'] = final_green_seconds.clip(lower=20, upper=99).round().astype(int)

    # 清理暫存欄位
    df = df.drop(columns=['weighted_flow', 'flow_ratio_yi', 'effective_green_webster'], errors='ignore')

    return df
