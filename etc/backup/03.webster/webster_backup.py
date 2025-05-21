def assign_green_seconds(df, n_phases=2, saturation_flow=1900, loss_time_per_phase=4):
    """
    結合 Webster 號誌配時理論與實時交通數據，計算並分配綠燈秒數。

    參數:
    df (pd.DataFrame): 包含交通數據的 DataFrame，應包含以下欄位：
                       'Volume_S' (小型車流量), 'Volume_L' (大型車流量), 'Volume_T' (拖車流量),
                       'Occupancy' (佔有率), 'Speed_S' (小型車速度), 'Speed_L' (大型車速度),
                       'Speed_T' (拖車速度), 'IsPeakHour' (是否為尖峰時段，布林值或 0/1)。
                       注意：Speed_S, Speed_L, Speed_T 應為數值，若有 NaN 需預先處理。
    n_phases (int): 路口時相數量 (例如，十字路口通常為 2)。
    saturation_flow (int): 每車道每小時的飽和流量 (veh/h)。
    loss_time_per_phase (int): 每個時相的損失時間 (秒)。

    回傳:
    pd.DataFrame: 包含新增 'green_seconds' 欄位的 DataFrame，代表最終分配的有效綠燈秒數。
    """

    # --- 數據預處理：處理潛在的 NaN 值，確保計算順利 ---
    # 針對可能影響數值計算的欄位，將 NaN 填補為 0 或其他合理值
    # 這裡選擇填補 0，因為流量、速度、佔有率為 NaN 通常意味著沒有數據或為零
    cols_to_check = ['Volume_S', 'Volume_L', 'Volume_T', 'Occupancy',
                     'Speed_S', 'Speed_L', 'Speed_T', 'IsPeakHour']
    for col in cols_to_check:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(0)

    # 確保 IsPeakHour 是數值型 (0 或 1)
    if 'IsPeakHour' in df.columns:
        df['IsPeakHour'] = df['IsPeakHour'].astype(int)

    # --- Webster 公式的計算部分 ---
    # 1. 計算總加權流量 (通常對應到時相中的關鍵流向流量)
    # 這裡假設 df 的每一列代表一個需要獨立計算綠燈時間的「方向」或「時相」
    # 因此我們對 df 的每一列計算其加權流量
    vehicle_weights_for_webster = {
        'Volume_S': 1.0, # 為 Webster 公式的流量比設定基礎權重
        'Volume_L': 1.5, # 大型車輛對容量影響較大
        'Volume_T': 2.0  # 拖車影響更大
    }

    # 確保所有參與計算的 Volume 欄位都存在，如果不存在則視為 0
    current_volumes = {}
    for k in ['Volume_S', 'Volume_L', 'Volume_T']:
        current_volumes[k] = df[k] if k in df.columns else 0

    df['weighted_flow'] = sum(current_volumes[k] * w for k, w in vehicle_weights_for_webster.items())

    # 2. 計算每個方向或時相的流量比 (y_i)
    # y_i = 實際流量 / 飽和流量
    # 避免 saturation_flow 為 0 導致除以零
    if saturation_flow == 0:
        # print("Warning: saturation_flow is 0, setting flow_ratio_yi to 0.")
        df['flow_ratio_yi'] = 0.0
    else:
        df['flow_ratio_yi'] = df['weighted_flow'] / saturation_flow

    # 3. 計算總流量比 (Y)
    # 這裡假設 total_flow_ratio 是所有關鍵時相的流量比之和。
    # 為了簡化，如果 df 的每一列是一個時相的數據，我們將所有時相的 y_i 加總。
    total_flow_ratio_Y = df['flow_ratio_yi'].sum()

    # 避免除以零或負值，確保 total_flow_ratio_Y 小於 1
    # 如果總流量比大於等於 1，表示需求已超過容量，無法依 Webster 公式計算，
    # 這裡將其限制在一個接近 1 的值，以避免運行錯誤。
    # 實際應用中，可能需要設定一個預設的週期或觸發過載模式。
    if total_flow_ratio_Y >= 1.0:
        total_flow_ratio_Y = 0.95 # 限制在 0.95，給予一些緩衝

    # 4. 計算總損失時間 (L)
    total_loss_time_L = n_phases * loss_time_per_phase

    # 5. 計算 Webster 公式的週期長度 (C)
    # C = (1.5 * L + 5) / (1 - Y)
    # 再次檢查分母，避免 total_flow_ratio_Y 接近 1 導致極大值
    denominator = (1 - total_flow_ratio_Y)
    if denominator <= 0.05: # 如果分母過小，表示路口接近飽和，週期會非常長
        # print("Warning: Denominator for cycle_length_C is too small, indicating near-saturation. Capped to max_cycle_length.")
        cycle_length_C = 120 # 直接設定為最大週期
    else:
        cycle_length_C = (1.5 * total_loss_time_L + 5) / denominator

    cycle_length_C = min(max(cycle_length_C, 30), 120) # 限制週期長度在 30-120 秒

    # 6. 計算每個方向或時相的基礎有效綠燈時間 (g_i)
    # g_i = (C - L) * (y_i / Y)
    # 避免 total_flow_ratio_Y 為 0 導致除以零
    if total_flow_ratio_Y == 0:
        # 如果總流量比為 0，表示沒有流量，基礎綠燈時間可以設為最小值或平均分配
        df['effective_green_webster'] = (cycle_length_C - total_loss_time_L) / n_phases if n_phases > 0 else 0
    else:
        df['effective_green_webster'] = (cycle_length_C - total_loss_time_L) * \
                                        (df['flow_ratio_yi'] / total_flow_ratio_Y)
    # 確保 effective_green_webster 不會是負值
    df['effective_green_webster'] = df['effective_green_webster'].clip(lower=0)


    # --- 您的動態調整邏輯 (作為對 Webster 基礎值的修正) ---
    # 1. 基本綠燈秒數（壅塞程度）
    # 確保 Occupancy 欄位存在且為數值
    base_adjustment = 20 + (df['Occupancy'] / 100.0) * 70 if 'Occupancy' in df.columns else 20

    # 2. 車輛數量加成（小中大型車不同權重）
    vehicle_weights_bonus = {
        'Volume_S': 2.0,
        'Volume_L': 2.5,
        'Volume_T': 3.0
    }
    current_volumes_bonus = {}
    for k in ['Volume_S', 'Volume_L', 'Volume_T']:
        current_volumes_bonus[k] = df[k] if k in df.columns else 0
    vehicle_bonus = sum(current_volumes_bonus[k] * w for k, w in vehicle_weights_bonus.items())

    # 3. 速度懲罰（如果平均車速過慢）
    # 確保所有 Speed 欄位都存在且為數值
    speeds = []
    for s_col in ['Speed_S', 'Speed_L', 'Speed_T']:
        if s_col in df.columns:
            speeds.append(df[s_col])
    if speeds:
        avg_vehicle_speed = sum(speeds) / len(speeds)
    else:
        avg_vehicle_speed = 100 # 如果沒有速度數據，假設速度很快，無懲罰

    speed_penalty = ((avg_vehicle_speed < 30).astype(int)) * 15

    # 4. 尖峰時段加成
    peak_bonus = df['IsPeakHour'].astype(int) * 20 if 'IsPeakHour' in df.columns else 0

    # --- 結合 Webster 基礎綠燈時間與動態調整 ---
    # 這裡將動態調整項作為對 Webster 計算出的基礎綠燈時間的額外增量
    final_green_seconds = df['effective_green_webster'] + \
                          base_adjustment + \
                          vehicle_bonus + \
                          speed_penalty + \
                          peak_bonus

    # 5. 限制綠燈秒數範圍：20 ~ 90 秒
    df['green_seconds'] = final_green_seconds.clip(lower=20, upper=90).round()

    # 清理中間計算欄位，保持回傳的 DataFrame 簡潔
    df = df.drop(columns=['weighted_flow', 'flow_ratio_yi', 'effective_green_webster'], errors='ignore')

    return df