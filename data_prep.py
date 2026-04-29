import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 [1/4] 데이터 로딩 시작...")
    
    # 1. 주기장(Zone) 데이터 로드
    file_zone = 'rksi_stands_zoned (2).csv' if os.path.exists('rksi_stands_zoned (2).csv') else 'rksi_stands_zoned.csv'
    df_zone = pd.read_csv(file_zone) if os.path.exists(file_zone) else pd.DataFrame()
    if not df_zone.empty: df_zone['Stand_ID'] = df_zone['Stand_ID'].astype(str)

    # 2. 항공편 데이터 로드
    flight_files = glob.glob('*RAMP*.csv') + glob.glob('*출발*.csv') + glob.glob('*도착*.csv')
    df_list = []
    for f in set(flight_files):
        try:
            try: temp_df = pd.read_csv(f, encoding='utf-8')
            except: temp_df = pd.read_csv(f, encoding='cp949')
            temp_df.rename(columns=lambda x: x.replace('癤풡', 'D').strip(), inplace=True)
            
            if '도착' in f or 'ARR' in f.upper(): temp_df['Flight_Dir'] = 'ARR'
            elif '출발' in f or 'DEP' in f.upper(): temp_df['Flight_Dir'] = 'DEP'
            else: temp_df['Flight_Dir'] = 'UNK'
            df_list.append(temp_df)
        except: pass
    
    if not df_list:
        print("🚨 원본 비행 데이터가 없습니다.")
        return
        
    df_flight = pd.concat(df_list, ignore_index=True)
    if 'STS' not in df_flight.columns: df_flight['STS'] = 'NML'
    df_flight['STS'] = df_flight['STS'].fillna('NML')
    df_flight['STS_Detail'] = df_flight['Flight_Dir'] + "_" + df_flight['STS']
    if 'SPT' in df_flight.columns: df_flight['SPT'] = df_flight['SPT'].astype(str)

    # 3. 기상 데이터 로드
    weather_files = glob.glob('AMOS_RKSI_*.csv') + glob.glob('기상_*.csv')
    w_list = []
    for f in set(weather_files):
        try:
            try: w_list.append(pd.read_csv(f, encoding='utf-8'))
            except: w_list.append(pd.read_csv(f, encoding='cp949'))
        except: pass
    df_weather = pd.concat(w_list, ignore_index=True) if w_list else pd.DataFrame()

    print("⏳ [2/4] 비행 시간 및 지연 계산 중 (자정 교차 보정)...")
    df_flight['Date_str'] = df_flight['Date'].astype(str)
    
    def parse_dt(date_str, time_str):
        try: 
            d_str, t_str = str(date_str).strip(), str(time_str).strip()
            if d_str.lower() == 'nan': return pd.NaT
            return pd.to_datetime(f"{d_str if len(d_str)==8 else '20'+d_str} {t_str}", format='%Y%m%d %H:%M')
        except: return pd.NaT

    df_flight['Plan_Time'] = df_flight.get('STD', pd.NaT)
    arr_mask = df_flight['Flight_Dir'] == 'ARR'
    if 'STA' in df_flight.columns: df_flight.loc[arr_mask, 'Plan_Time'] = df_flight.loc[arr_mask, 'STA']
    
    df_flight['STD_Full'] = df_flight.apply(lambda x: parse_dt(x['Date_str'], x['Plan_Time']), axis=1)
    df_flight = df_flight.dropna(subset=['STD_Full']) 
    
    def calc_all_times(row):
        std = row['STD_Full']
        def adjust_time_crossing(base_dt, target_time_str):
            if pd.isna(target_time_str): return pd.NaT
            try:
                t = pd.to_datetime(str(target_time_str), format='%H:%M').time()
                target_dt = base_dt.replace(hour=t.hour, minute=t.minute)
                diff = (target_dt - base_dt).total_seconds()
                if diff < -43200: target_dt += timedelta(days=1)
                elif diff > 43200: target_dt -= timedelta(days=1)
                return target_dt
            except: return pd.NaT

        ram_dt = adjust_time_crossing(std, row.get('RAM'))
        act_col = 'ATD' if row['Flight_Dir'] == 'DEP' else 'ATA'
        if act_col not in row: act_col = 'ATD' if 'ATD' in row else ('ATA' if 'ATA' in row else None)
        act_dt = adjust_time_crossing(ram_dt if not pd.isna(ram_dt) else std, row.get(act_col))

        taxi_time, total_delay = 0.0, 0.0
        if not pd.isna(act_dt): total_delay = (act_dt - std).total_seconds() / 60.0
            
        if 'ATD-RAM' in row and pd.notna(row['ATD-RAM']): taxi_time = float(row['ATD-RAM'])
        elif 'RAM-ATA' in row and pd.notna(row['RAM-ATA']): taxi_time = float(row['RAM-ATA'])
        elif not pd.isna(act_dt) and not pd.isna(ram_dt): taxi_time = abs((act_dt - ram_dt).total_seconds() / 60.0)
             
        return ram_dt, act_dt, taxi_time, total_delay

    res = df_flight.apply(calc_all_times, axis=1, result_type='expand')
    df_flight['RAM_Full'] = res[0]
    df_flight['ATD_Full'] = res[1]
    df_flight['Taxi_Time'] = res[2]
    df_flight['Total_Delay'] = res[3]
    
    df_flight['Taxi_Out'] = np.where(df_flight['Flight_Dir'] == 'DEP', df_flight['Taxi_Time'], np.nan)
    df_flight['Taxi_In'] = np.where(df_flight['Flight_Dir'] == 'ARR', df_flight['Taxi_Time'], np.nan)
    
    df_flight['YM'] = df_flight['STD_Full'].dt.to_period('M').astype(str)
    df_flight['Date_Only'] = df_flight['STD_Full'].dt.date.astype(str)
    df_flight['Hour'] = df_flight['STD_Full'].dt.hour
    
    df_flight['Is_Delayed'] = (df_flight['Total_Delay'] > 15) | (df_flight['STS'].isin(['DLA', 'CNL']))
    
    valid_taxi = df_flight[df_flight['Taxi_Time'] > 0]
    stats = valid_taxi.groupby('YM')['Taxi_Time'].agg(['mean', 'std']).reset_index()
    stats['Limit_1Sigma'] = stats['mean'] + stats['std']
    df_flight = pd.merge(df_flight, stats, on='YM', how='left')
    
    def classify_delay_stat(row):
        if row['STS'] == 'CNL': return 'Cancelled (CNL)'
        if not row['Is_Delayed']: return 'Normal'
        limit = row['Limit_1Sigma'] if pd.notna(row['Limit_1Sigma']) else 30
        if row['Taxi_Time'] >= limit: return 'Taxi (Ground)'
        else: return 'Ramp (Gate)'
    df_flight['Delay_Cause'] = df_flight.apply(classify_delay_stat, axis=1)

    print("☁️ [3/4] 기상 데이터 및 주기장 매핑 중...")
    df_merged = pd.merge(df_flight, df_zone, left_on='SPT', right_on='Stand_ID', how='left') if not df_zone.empty else df_flight

    if not df_weather.empty:
        df_weather['DT'] = pd.to_datetime(df_weather['일시'], errors='coerce')
        df_weather = df_weather.dropna(subset=['DT'])
        df_weather = df_weather.rename(columns={'기온(°C)': 'Temp', '이슬점온도(°C)': 'Dew_Point', '풍속(KT)': 'Wind_Spd', '풍향(deg)': 'Wind_Dir', '시정(m)': 'Visibility', '강수량(mm)': 'Precip', '일기현상': 'W_Code'})
        df_weather['Precip'] = df_weather['Precip'].fillna(0)
        df_weather['Visibility'] = df_weather['Visibility'].fillna(10000)
        df_weather['Dew_Point'] = df_weather.get('Dew_Point', np.nan)
        
        def parse_wmo_code(code):
            if pd.isna(code): return "기상현상 없음"
            try: c = int(code)
            except: return "기타"
            if 36 <= c <= 39: return "날림눈/눈보라"
            elif 56 <= c <= 57 or 66 <= c <= 67: return "어는 비(결빙)"
            elif 68 <= c <= 79 or 83 <= c <= 86: return "강설/진눈깨비"
            elif 40 <= c <= 49: return "안개/빙무"
            elif 60 <= c <= 65 or 80 <= c <= 82: return "비/소나기"
            else: return "일반"
            
        df_weather['Weather_Desc'] = df_weather['W_Code'].apply(parse_wmo_code)
        
        df_weather['Hour_DT'] = df_weather['DT'].dt.floor('H')
        df_merged['Hour_DT'] = df_merged['STD_Full'].dt.floor('H')
        hourly_w_desc = df_weather.groupby('Hour_DT').first().reset_index()
        weather_cols = ['Hour_DT', 'Temp', 'Dew_Point', 'Visibility', 'Wind_Spd', 'Wind_Dir', 'Precip', 'Weather_Desc']
        df_merged = pd.merge(df_merged, hourly_w_desc[[c for c in weather_cols if c in hourly_w_desc.columns]], on='Hour_DT', how='left')
        
        daily_weather = df_weather.groupby(df_weather['DT'].dt.date)['Weather_Desc'].apply(
            lambda x: '강설_결빙(De-icing)' if any(k in w for w in x for k in ['눈', '설', '빙', '결빙', '진눈깨비']) else '일반'
        ).reset_index()
        daily_weather['Date_Only'] = daily_weather['DT'].astype(str)
        df_merged = pd.merge(df_merged, daily_weather[['Date_Only', 'Snow_Status']], on='Date_Only', how='left')
        df_merged['Snow_Status'] = df_merged['Snow_Status'].fillna('일반')
    else:
        df_merged['Snow_Status'] = '일반'
        df_merged['Weather_Desc'] = '-'
        for c in ['Temp', 'Dew_Point', 'Visibility', 'Wind_Spd', 'Wind_Dir', 'Precip']: df_merged[c] = np.nan

    print("💾 [4/4] 대시보드 전용 경량화 파일 저장 중...")
    # 시각화에 필요 없는 수많은 원본 컬럼을 버리고 메모리를 극한으로 절약합니다.
    final_cols = ['FLT', 'Flight_Dir', 'STS_Detail', 'YM', 'Date_Only', 'Hour', 'STD_Full', 'RAM_Full', 
                  'Total_Delay', 'Taxi_Out', 'Taxi_In', 'Is_Delayed', 'Delay_Cause', 
                  'Lat', 'Lon', 'Snow_Status', 'Temp', 'Dew_Point', 'Visibility', 'Wind_Spd', 'Wind_Dir', 'Precip', 'Weather_Desc']
    
    df_final = df_merged[[c for c in final_cols if c in df_merged.columns]]
    
    # CSV 저장 (대시보드 앱에서 읽을 파일)
    df_final.to_csv('master_dashboard_data.csv', index=False, encoding='utf-8-sig')
    print("✅ 완료! 'master_dashboard_data.csv'가 생성되었습니다.")

if __name__ == "__main__":
    main()
