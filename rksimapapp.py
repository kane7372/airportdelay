import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import altair as alt
import os
import glob
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Incheon Airport Ultimate Flight & Weather", layout="wide")

# ==========================================
# 1. Load Data
# ==========================================
@st.cache_data
def load_data():
    file_zone = 'rksi_stands_zoned (2).csv'
    if not os.path.exists(file_zone):
        if os.path.exists('rksi_stands_zoned.csv'): file_zone = 'rksi_stands_zoned.csv'
        else: return None, None, None, None, "Zone file not found"
            
    df_zone = pd.read_csv(file_zone)
    df_zone['Stand_ID'] = df_zone['Stand_ID'].astype(str)

    # Load Flights
    flight_files = glob.glob('*RAMP*.csv') + glob.glob('*출발*.csv') + glob.glob('*도착*.csv')
    df_list = []
    for f in set(flight_files):
        try: 
            temp_df = pd.read_csv(f)
            # 파일 이름에 '도착'이나 'ARR'이 있으면 확실하게 ARR로 지정
            if '도착' in f or 'ARR' in f.upper():
                temp_df['STS'] = 'ARR'
            # 파일 이름에 '출발'이나 'DEP'가 있으면 확실하게 DEP로 지정
            elif '출발' in f or 'DEP' in f.upper():
                temp_df['STS'] = 'DEP'
            df_list.append(temp_df)
        except: pass
    if not df_list: return None, None, None, None, "No Flight files found"
    df_flight = pd.concat(df_list, ignore_index=True)
    
    # Load Weather
    weather_files = glob.glob('AMOS_RKSI_*.csv') + glob.glob('기상_*.csv')
    df_weather = pd.DataFrame()
    w_list = []
    for f in set(weather_files):
        try:
            try: w_list.append(pd.read_csv(f, encoding='utf-8'))
            except: w_list.append(pd.read_csv(f, encoding='cp949'))
        except: pass
    if w_list: df_weather = pd.concat(w_list, ignore_index=True)

    # --- Preprocessing ---
    if 'SPT' in df_flight.columns: df_flight['SPT'] = df_flight['SPT'].astype(str)
    if 'STS' not in df_flight.columns: df_flight['STS'] = 'DEP' # Default if missing
    df_flight['Date_str'] = df_flight['Date'].astype(str)
    
    def parse_dt(date_str, time_str):
        try: 
            d_str, t_str = str(date_str).strip(), str(time_str).strip()
            return pd.to_datetime(f"{d_str if len(d_str)==8 else '20'+d_str} {t_str}", format='%Y%m%d %H:%M')
        except: return pd.NaT

    df_flight['Plan_Time'] = df_flight['STD'] if 'STD' in df_flight.columns else df_flight.get('STA', pd.NaT)
    df_flight['STD_Full'] = df_flight.apply(lambda x: parse_dt(x['Date_str'], x['Plan_Time']), axis=1)
    
    def calc_all_times(row):
        std = row['STD_Full']
        if pd.isna(std): return pd.NaT, pd.NaT, 0, 0, 0
        
        ram_dt = pd.NaT
        try:
            if pd.notna(row.get('RAM')):
                ram_time = pd.to_datetime(str(row['RAM']), format='%H:%M').time()
                ram_dt = std.replace(hour=ram_time.hour, minute=ram_time.minute)
                if std.hour < 4 and ram_dt.hour > 20: ram_dt -= timedelta(days=1)
                elif std.hour > 20 and ram_dt.hour < 4: ram_dt += timedelta(days=1)
        except: pass

        act_dt = pd.NaT
        act_col = 'ATD' if 'ATD' in row and pd.notna(row['ATD']) else ('ATA' if 'ATA' in row and pd.notna(row['ATA']) else None)
        try:
            if act_col:
                act_time = pd.to_datetime(str(row[act_col]), format='%H:%M').time()
                base_dt = ram_dt if not pd.isna(ram_dt) else std
                act_dt = base_dt.replace(hour=act_time.hour, minute=act_time.minute)
                if act_dt < base_dt: act_dt += timedelta(days=1)
        except: pass

        ramp_delay, taxi_time, total_delay = 0, 0, 0
        if not pd.isna(act_dt): total_delay = abs((act_dt - std).total_seconds() / 60)
        if not pd.isna(ram_dt): ramp_delay = abs((ram_dt - std).total_seconds() / 60)
            
        if 'ATD-RAM' in row and pd.notna(row['ATD-RAM']): taxi_time = float(row['ATD-RAM'])
        elif 'RAM-ATA' in row and pd.notna(row['RAM-ATA']): taxi_time = float(row['RAM-ATA'])
        elif not pd.isna(act_dt) and not pd.isna(ram_dt): taxi_time = abs((act_dt - ram_dt).total_seconds() / 60)
             
        return ram_dt, act_dt, ramp_delay, taxi_time, total_delay

    res = df_flight.apply(calc_all_times, axis=1, result_type='expand')
    df_flight['RAM_Full'] = res[0]
    df_flight['ATD_Full'] = res[1]
    df_flight['Ramp_Delay'] = res[2]
    df_flight['Taxi_Time'] = res[3]
    df_flight['Total_Delay'] = res[4]
    
    df_flight['YM'] = df_flight['STD_Full'].dt.to_period('M').astype(str)
    df_flight['Date_Only'] = df_flight['STD_Full'].dt.date
    df_flight['Hour'] = df_flight['STD_Full'].dt.hour
    df_flight['Is_Delayed'] = df_flight['Total_Delay'] > 15
    
    valid_taxi = df_flight[df_flight['Taxi_Time'] > 0]
    stats = valid_taxi.groupby('YM')['Taxi_Time'].agg(['mean', 'std']).reset_index()
    stats['Limit_1Sigma'] = stats['mean'] + stats['std']
    
    df_flight = pd.merge(df_flight, stats, on='YM', how='left')
    
    def classify_delay_stat(row):
        if not row['Is_Delayed']: return 'Normal'
        limit = row['Limit_1Sigma'] if pd.notna(row['Limit_1Sigma']) else 30
        if row['Taxi_Time'] >= limit: return 'Taxi (Ground)'
        else: return 'Ramp (Gate)'

    df_flight['Delay_Cause'] = df_flight.apply(classify_delay_stat, axis=1)
    df_merged = pd.merge(df_flight, df_zone, left_on='SPT', right_on='Stand_ID', how='inner') if 'SPT' in df_flight.columns else df_flight

    # --- Weather ---
    if not df_weather.empty:
        df_weather['DT'] = pd.to_datetime(df_weather['일시'])
        df_weather = df_weather.rename(columns={'기온(°C)': 'Temp', '풍속(KT)': 'Wind_Spd', '풍향(deg)': 'Wind_Dir', '시정(m)': 'Visibility', '강수량(mm)': 'Precip', '일기현상': 'W_Code'})
        df_weather['Precip'] = df_weather['Precip'].fillna(0)
        df_weather['Visibility'] = df_weather['Visibility'].fillna(10000)
        
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
        daily_weather = df_weather.groupby(df_weather['DT'].dt.date)['Weather_Desc'].apply(
            lambda x: '강설/결빙 (De-icing)' if any(k in w for w in x for k in ['눈', '설', '빙', '결빙', '진눈깨비']) else '일반'
        ).reset_index()
        daily_weather.columns = ['Date_Only', 'Snow_Status']
        
        df_merged = pd.merge(df_merged, daily_weather, on='Date_Only', how='left')
        df_merged['Snow_Status'] = df_merged['Snow_Status'].fillna('일반')
        
    return df_merged, df_weather, stats, df_zone, "Success"

flights, weather, taxi_stats, zone_data, msg = load_data()

if flights is None:
    st.error(msg)
    st.stop()

# ==========================================
# 2. UI Layout - 세분화된 4개 탭 구성
# ==========================================
st.title("🛫 인천공항 통계 기반 지연 분석 (Top-Down)")

tab1, tab2, tab3, tab4 = st.tabs([
    "📅 1. 월별 통계 (Macro)", 
    "📆 2. 일별 통계 (Daily Trends)", 
    "⏰ 3. 시간대별 통계 (Micro)", 
    "🗺️ 4. 상세 지도 분석 (Location)"
])

# ------------------------------------------
# [TAB 1] 월별 통계 (Monthly)
# ------------------------------------------
with tab1:
    st.header("📅 월별 운항 및 지연 트렌드")
    
    # 통계 집계 (월별, 출/도착별)
    monthly_stats = flights.groupby(['YM', 'STS']).agg(
        Flight_Count=('FLT', 'count'),
        Delay_Count=('Is_Delayed', 'sum'),
        Avg_Delay_Time=('Total_Delay', lambda x: x[x > 15].mean() if len(x[x > 15]) > 0 else 0),
        Avg_Taxi_Time=('Taxi_Time', 'mean')
    ).reset_index()
    
    monthly_stats['Delay_Rate (%)'] = (monthly_stats['Delay_Count'] / monthly_stats['Flight_Count']) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(monthly_stats, x='YM', y='Flight_Count', color='STS', barmode='group', title="월별 출/도착 운항 편수", text_auto=True)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.line(monthly_stats, x='YM', y='Delay_Count', color='STS', markers=True, title="월별 지연(>15분) 발생 건수")
        st.plotly_chart(fig2, use_container_width=True)
        
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.bar(monthly_stats, x='YM', y='Avg_Delay_Time', color='STS', barmode='group', title="지연 항공편의 월평균 지연 시간(분)")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.line(monthly_stats, x='YM', y='Avg_Taxi_Time', color='STS', markers=True, title="월평균 지상 이동시간 (Taxi Time)")
        st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------
# [TAB 2] 일별 통계 (Daily)
# ------------------------------------------
with tab2:
    st.header("📆 일별 운항 및 지연 트렌드")
    
    daily_stats = flights.groupby(['Date_Only', 'STS', 'Snow_Status']).agg(
        Flight_Count=('FLT', 'count'),
        Delay_Count=('Is_Delayed', 'sum'),
        Avg_Taxi_Time=('Taxi_Time', 'mean')
    ).reset_index()
    
    # 기상 현상(강설/일반)에 따른 지상 이동시간 비교
    st.subheader("❄️ 강설 여부에 따른 일별 평균 지상이동시간")
    fig_daily_taxi = px.line(daily_stats, x='Date_Only', y='Avg_Taxi_Time', color='STS', facet_row='Snow_Status',
                             markers=True, height=600)
    fig_daily_taxi.update_yaxes(matches=None) # 각 행의 Y축을 독립적으로 설정하여 차이 부각
    st.plotly_chart(fig_daily_taxi, use_container_width=True)
    
    st.subheader("📊 일별 지연 건수 추이")
    fig_daily_delay = px.bar(daily_stats, x='Date_Only', y='Delay_Count', color='STS', title="일별 지연 항공편 수")
    st.plotly_chart(fig_daily_delay, use_container_width=True)

# ------------------------------------------
# [TAB 3] 시간대별 통계 (Hourly)
# ------------------------------------------
with tab3:
    st.header("⏰ 시간대별(0~23시) 병목 현상 분석")
    st.markdown("특정 날짜 범위를 선택하여 **하루 중 어느 시간대**에 지연과 주기장 체증이 발생하는지 확인합니다.")
    
    # 기간 필터
    min_date, max_date = flights['Date_Only'].min(), flights['Date_Only'].max()
    sel_date_range = st.date_input("분석 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(sel_date_range) == 2:
        start_d, end_d = sel_date_range
        # Date_Only가 비어있는(NaN/NaT) 에러 유발 데이터 제거 후 필터링
        valid_dates_df = flights.dropna(subset=['Date_Only'])
        filtered_hourly = valid_dates_df[(valid_dates_df['Date_Only'] >= start_d) & (valid_dates_df['Date_Only'] <= end_d)]
        
        hourly_stats = filtered_hourly.groupby(['Hour', 'STS']).agg(
            Flight_Count=('FLT', 'count'),
            Delay_Count=('Is_Delayed', 'sum'),
            Avg_Taxi_Time=('Taxi_Time', 'mean')
        ).reset_index()
        
        c1, c2 = st.columns(2)
        with c1:
            fig_h1 = px.bar(hourly_stats, x='Hour', y='Flight_Count', color='STS', barmode='group', 
                            title="시간대별 출/도착 스케줄 집중도 (Flight Count)")
            fig_h1.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            st.plotly_chart(fig_h1, use_container_width=True)
            
        with c2:
            fig_h2 = px.line(hourly_stats, x='Hour', y='Avg_Taxi_Time', color='STS', markers=True, 
                             title="시간대별 평균 지상 이동시간 (Taxi Time bottleneck)")
            fig_h2.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            st.plotly_chart(fig_h2, use_container_width=True)

# ------------------------------------------
# [TAB 4] 상세 지도 분석 (기존 기능)
# ------------------------------------------
with tab4:
    st.header("🗺️ 상세 지연 인과 및 지도 시각화")
    st.sidebar.header("지도 세부 설정 (Tab 4 전용)")
    sel_date = st.sidebar.date_input("지도 표시 날짜", min_date, min_value=min_date, max_value=max_date)
    sel_hour = st.sidebar.slider("지도 표시 시간", 0, 23, 12)
    time_basis = st.sidebar.radio("지도 기준 시간", ["STD (계획)", "RAM (푸시백)"], index=1)
    
    target_col = "STD_Full" if time_basis == "STD (계획)" else "RAM_Full"
    valid_flights = flights.dropna(subset=[target_col]).copy()
    map_flights = valid_flights[(valid_flights[target_col].dt.date == sel_date) & (valid_flights[target_col].dt.hour == sel_hour)].copy()

    # (이하 기존 지도 출력 로직과 동일하게 작성)
    c1, c2, c3 = st.columns(3)
    c1.metric("해당 시간 편수", f"{len(map_flights)}편")
    
    if 'Lat' in map_flights.columns:
        m = folium.Map(location=[37.46, 126.44], zoom_start=13)
        # 활주로
        runways = {'33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363), '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)}
        for r, c in runways.items(): folium.Marker(c, popup=r, icon=folium.Icon(color='gray', icon='plane')).add_to(m)

        # 항공기 마커
        color_dict = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange'}
        for _, row in map_flights.iterrows():
            color = color_dict.get(row['Delay_Cause'], 'blue')
            popup = f"<b>{row['FLT']} ({row['STS']})</b><br>Delay: {row['Delay_Cause']}<br>Taxi: {row['Taxi_Time']:.1f}m<br>Total: {row['Total_Delay']:.0f}m"
            folium.Marker([row['Lat'], row['Lon']], popup=popup, tooltip=f"{row['FLT']}", icon=folium.Icon(color=color, icon='plane')).add_to(m)

        st_folium(m, width="100%", height=600)
    else:
        st.warning("지도 시각화를 위한 주기장 좌표 데이터가 부족합니다.")
