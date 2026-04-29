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
from plotly.subplots import make_subplots

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

    # --- 항공편 데이터 로드 ---
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
        except Exception as e:
            pass
        
    if not df_list: return None, None, None, None, "No Flight files found"
    df_flight = pd.concat(df_list, ignore_index=True)
    
    if 'STS' not in df_flight.columns: df_flight['STS'] = 'NML'
    df_flight['STS'] = df_flight['STS'].fillna('NML')
    df_flight['STS_Detail'] = df_flight['Flight_Dir'] + "_" + df_flight['STS']

    # --- 기상 데이터 로드 ---
    weather_files = glob.glob('AMOS_RKSI_*.csv') + glob.glob('기상_*.csv')
    df_weather = pd.DataFrame()
    w_list = []
    for f in set(weather_files):
        try:
            try: w_list.append(pd.read_csv(f, encoding='utf-8'))
            except: w_list.append(pd.read_csv(f, encoding='cp949'))
        except: pass
    if w_list: df_weather = pd.concat(w_list, ignore_index=True)

    # --- 데이터 전처리 ---
    if 'SPT' in df_flight.columns: df_flight['SPT'] = df_flight['SPT'].astype(str)
    df_flight['Date_str'] = df_flight['Date'].astype(str)
    
    def parse_dt(date_str, time_str):
        try: 
            d_str, t_str = str(date_str).strip(), str(time_str).strip()
            if d_str.lower() == 'nan': return pd.NaT
            return pd.to_datetime(f"{d_str if len(d_str)==8 else '20'+d_str} {t_str}", format='%Y%m%d %H:%M')
        except: return pd.NaT

    df_flight['Plan_Time'] = df_flight.get('STD', pd.NaT)
    arr_mask = df_flight['Flight_Dir'] == 'ARR'
    if 'STA' in df_flight.columns:
        df_flight.loc[arr_mask, 'Plan_Time'] = df_flight.loc[arr_mask, 'STA']
    
    df_flight['STD_Full'] = df_flight.apply(lambda x: parse_dt(x['Date_str'], x['Plan_Time']), axis=1)
    df_flight = df_flight.dropna(subset=['STD_Full']) 
    
    def calc_all_times(row):
        std = row['STD_Full']
        if pd.isna(std): return pd.NaT, pd.NaT, 0, 0, 0
        
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
        if act_col not in row:
            act_col = 'ATD' if 'ATD' in row else ('ATA' if 'ATA' in row else None)
            
        act_dt = adjust_time_crossing(ram_dt if not pd.isna(ram_dt) else std, row.get(act_col))

        ramp_delay, taxi_time, total_delay = 0.0, 0.0, 0.0
        
        if not pd.isna(act_dt): total_delay = (act_dt - std).total_seconds() / 60.0
        if not pd.isna(ram_dt): ramp_delay = (ram_dt - std).total_seconds() / 60.0
            
        if 'ATD-RAM' in row and pd.notna(row['ATD-RAM']): taxi_time = float(row['ATD-RAM'])
        elif 'RAM-ATA' in row and pd.notna(row['RAM-ATA']): taxi_time = float(row['RAM-ATA'])
        elif not pd.isna(act_dt) and not pd.isna(ram_dt): taxi_time = abs((act_dt - ram_dt).total_seconds() / 60.0)
             
        return ram_dt, act_dt, ramp_delay, taxi_time, total_delay

    res = df_flight.apply(calc_all_times, axis=1, result_type='expand')
    df_flight['RAM_Full'] = res[0]
    df_flight['ATD_Full'] = res[1]
    df_flight['Ramp_Delay'] = res[2]
    df_flight['Taxi_Time'] = res[3]
    df_flight['Total_Delay'] = res[4]
    
    df_flight['Taxi_Out'] = np.where(df_flight['Flight_Dir'] == 'DEP', df_flight['Taxi_Time'], np.nan)
    df_flight['Taxi_In'] = np.where(df_flight['Flight_Dir'] == 'ARR', df_flight['Taxi_Time'], np.nan)
    
    df_flight['YM'] = df_flight['STD_Full'].dt.to_period('M').astype(str)
    df_flight['Date_Only'] = df_flight['STD_Full'].dt.date
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
    df_merged = pd.merge(df_flight, df_zone, left_on='SPT', right_on='Stand_ID', how='inner') if 'SPT' in df_flight.columns else df_flight

    # --- 기상 데이터 상세 매핑 ---
    if not df_weather.empty:
        df_weather['DT'] = pd.to_datetime(df_weather['일시'], errors='coerce')
        df_weather = df_weather.dropna(subset=['DT'])
        
        # [추가] 분석에 필요한 모든 기상 인자 컬럼화
        df_weather = df_weather.rename(columns={
            '기온(°C)': 'Temp', 
            '이슬점온도(°C)': 'Dew_Point',
            '풍속(KT)': 'Wind_Spd', 
            '풍향(deg)': 'Wind_Dir', 
            '순간풍속(KT)': 'Gust',
            '시정(m)': 'Visibility', 
            '강수량(mm)': 'Precip', 
            '일기현상': 'W_Code'
        })
        df_weather['Precip'] = df_weather['Precip'].fillna(0)
        df_weather['Visibility'] = df_weather['Visibility'].fillna(10000)
        df_weather['Dew_Point'] = df_weather.get('Dew_Point', np.nan) # 없으면 NaN
        
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
        
        # 병합 시 이슬점, 풍속 등 모두 포함
        hourly_w_desc = df_weather.groupby('Hour_DT').first().reset_index()
        weather_cols = ['Hour_DT', 'Temp', 'Dew_Point', 'Visibility', 'Wind_Spd', 'Wind_Dir', 'Precip', 'Weather_Desc']
        df_merged = pd.merge(df_merged, hourly_w_desc[[c for c in weather_cols if c in hourly_w_desc.columns]], on='Hour_DT', how='left')
        
        daily_weather = df_weather.groupby(df_weather['DT'].dt.date)['Weather_Desc'].apply(
            lambda x: '강설_결빙(De-icing)' if any(k in w for w in x for k in ['눈', '설', '빙', '결빙', '진눈깨비']) else '일반'
        ).reset_index()
        daily_weather.columns = ['Date_Only', 'Snow_Status']
        
        df_merged = pd.merge(df_merged, daily_weather, on='Date_Only', how='left', suffixes=('', '_y'))
        df_merged['Snow_Status'] = df_merged['Snow_Status'].fillna('일반')
    else:
        df_merged['Snow_Status'] = '일반'
        df_merged['Weather_Desc'] = '-'
        for c in ['Temp', 'Dew_Point', 'Visibility', 'Wind_Spd', 'Wind_Dir', 'Precip']:
            df_merged[c] = np.nan
        
    return df_merged, df_weather, stats, df_zone, "Success"

flights, weather, taxi_stats, zone_data, msg = load_data()

if flights is None:
    st.error(msg)
    st.stop()

# ==========================================
# 2. UI Layout
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
    monthly_stats = flights.groupby(['YM', 'STS_Detail']).agg(
        Flight_Count=('FLT', 'count'),
        Delay_Count=('Is_Delayed', 'sum'),
        Avg_Delay_Time=('Total_Delay', lambda x: x[x > 15].mean() if len(x[x > 15]) > 0 else 0),
        Avg_Taxi_Out=('Taxi_Out', 'mean'),
        Avg_Taxi_In=('Taxi_In', 'mean')
    ).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(monthly_stats, x='YM', y='Flight_Count', color='STS_Detail', barmode='stack', title="월별 출/도착 운항 편수 (상태별)")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.line(monthly_stats, x='YM', y='Delay_Count', color='STS_Detail', markers=True, title="월별 지연/결항 발생 건수")
        st.plotly_chart(fig2, use_container_width=True)
        
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.bar(monthly_stats, x='YM', y='Avg_Delay_Time', color='STS_Detail', barmode='group', title="지연 항공편의 월평균 지연 시간(분)")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        melted_taxi = monthly_stats.melt(id_vars=['YM', 'STS_Detail'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time')
        melted_taxi = melted_taxi.dropna(subset=['Time'])
        fig4 = px.line(melted_taxi, x='YM', y='Time', color='STS_Detail', line_dash='Taxi_Type', markers=True, title="월평균 지상이동시간 (Taxi-Out/In 분리)")
        st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------
# [TAB 2] 일별 통계 (Daily)
# ------------------------------------------
with tab2:
    st.header("📆 일별 운항 및 지연 트렌드")
    daily_stats = flights.groupby(['Date_Only', 'STS_Detail', 'Snow_Status']).agg(
        Flight_Count=('FLT', 'count'),
        Delay_Count=('Is_Delayed', 'sum'),
        Avg_Taxi_Out=('Taxi_Out', 'mean'),
        Avg_Taxi_In=('Taxi_In', 'mean')
    ).reset_index()
    
    st.subheader("❄️ 강설 여부에 따른 일별 평균 지상이동시간")
    melted_daily = daily_stats.melt(id_vars=['Date_Only', 'STS_Detail', 'Snow_Status'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time')
    melted_daily = melted_daily.dropna(subset=['Time'])
    
    fig_daily_taxi = px.line(melted_daily, x='Date_Only', y='Time', color='STS_Detail', line_dash='Taxi_Type', facet_row='Snow_Status', markers=True, height=600)
    fig_daily_taxi.update_yaxes(matches=None)
    st.plotly_chart(fig_daily_taxi, use_container_width=True)
    
    st.subheader("📊 일별 지연/결항 건수 추이")
    fig_daily_delay = px.bar(daily_stats, x='Date_Only', y='Delay_Count', color='STS_Detail', title="일별 지연 및 결항 건수", barmode='stack')
    st.plotly_chart(fig_daily_delay, use_container_width=True)

# ------------------------------------------
# [TAB 3] 시간대별 통계 (Hourly) + [추가] 맞춤형 기상 연동 차트
# ------------------------------------------
with tab3:
    st.header("⏰ 시간대별(0~23시) 병목 현상 및 종합 기상 연동 분석")
    min_date, max_date = flights['Date_Only'].min(), flights['Date_Only'].max()
    sel_date_range = st.date_input("분석 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(sel_date_range) == 2:
        start_d, end_d = sel_date_range
        filtered_hourly = flights[(flights['Date_Only'] >= start_d) & (flights['Date_Only'] <= end_d)]
        
        hourly_stats = filtered_hourly.groupby(['Hour', 'STS_Detail']).agg(
            Flight_Count=('FLT', 'count'),
            Delay_Count=('Is_Delayed', 'sum'),
            Avg_Taxi_Out=('Taxi_Out', 'mean'),
            Avg_Taxi_In=('Taxi_In', 'mean')
        ).reset_index()
        
        # 해당 기간의 기상 데이터 요약
        w_hourly = filtered_hourly.groupby('Hour').agg(
            Avg_Temp=('Temp', 'mean'),
            Avg_Dew=('Dew_Point', 'mean'),
            Avg_Vis=('Visibility', 'mean'),
            Avg_Wind=('Wind_Spd', 'mean'),
            Avg_Precip=('Precip', 'mean')
        ).reset_index()
        
        c1, c2 = st.columns(2)
        with c1:
            fig_h1 = px.bar(hourly_stats, x='Hour', y='Flight_Count', color='STS_Detail', barmode='stack', title="시간대별 출/도착 스케줄 집중도")
            fig_h1.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            st.plotly_chart(fig_h1, use_container_width=True)
            
        with c2:
            melted_h_taxi = hourly_stats.melt(id_vars=['Hour', 'STS_Detail'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time')
            melted_h_taxi = melted_h_taxi.dropna(subset=['Time'])
            fig_h2 = px.line(melted_h_taxi, x='Hour', y='Time', color='STS_Detail', line_dash='Taxi_Type', markers=True, title="시간대별 평균 지상 이동시간 (Out/In)")
            fig_h2.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            st.plotly_chart(fig_h2, use_container_width=True)
            
        # ---------------------------------------------
        # [핵심] 사용자가 선택 가능한 기상 다중 차트
        # ---------------------------------------------
        st.divider()
        st.subheader("🌤️ 맞춤형 시간대별 기상 차트")
        st.markdown("관측하고 싶은 기상 요소를 다중 선택하세요.")
        
        weather_options = {
            "기온 (°C)": "Avg_Temp",
            "이슬점 온도 (°C)": "Avg_Dew",
            "시정 (m)": "Avg_Vis",
            "풍속 (KT)": "Avg_Wind",
            "강수량 (mm)": "Avg_Precip"
        }
        selected_weather = st.multiselect(
            "기상 지표 선택",
            options=list(weather_options.keys()),
            default=["기온 (°C)", "이슬점 온도 (°C)", "시정 (m)"]
        )

        if selected_weather:
            # Create subplots with secondary y-axis for proper scaling
            fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
            
            colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
            
            for idx, w_name in enumerate(selected_weather):
                col_name = weather_options[w_name]
                # 시정이나 강수량 등 단위/스케일이 확연히 다른 지표는 오른쪽 Y축(Secondary) 사용
                is_secondary = w_name in ["시정 (m)", "강수량 (mm)"]
                
                # 강수량은 Bar, 나머지는 Line
                if w_name == "강수량 (mm)":
                    fig_weather.add_trace(go.Bar(x=w_hourly['Hour'], y=w_hourly[col_name], name=w_name, marker_color=colors[idx], opacity=0.6), secondary_y=is_secondary)
                else:
                    fig_weather.add_trace(go.Scatter(x=w_hourly['Hour'], y=w_hourly[col_name], name=w_name, mode='lines+markers', line=dict(color=colors[idx])), secondary_y=is_secondary)
            
            fig_weather.update_layout(title="선택된 지표의 시간대별 평균 트렌드", hovermode="x unified")
            fig_weather.update_xaxes(title_text="시간대 (Hour)", tickmode='linear', tick0=0, dtick=1)
            fig_weather.update_yaxes(title_text="온도/풍속 등", secondary_y=False)
            fig_weather.update_yaxes(title_text="시정/강수량", secondary_y=True)
            
            st.plotly_chart(fig_weather, use_container_width=True)
        else:
            st.info("차트로 보고 싶은 기상 지표를 위에서 선택해 주세요.")

# ------------------------------------------
# [TAB 4] 상세 지도 분석
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("해당 시간 편수", f"{len(map_flights)}편")
    
    if not map_flights.empty:
        # 단일 시간 기상 상세 표시 (풍속, 풍향 포함)
        w_desc = map_flights['Weather_Desc'].iloc[0] if 'Weather_Desc' in map_flights.columns else '-'
        t_val = map_flights['Temp'].iloc[0] if 'Temp' in map_flights.columns else '-'
        dew_val = map_flights['Dew_Point'].iloc[0] if 'Dew_Point' in map_flights.columns else '-'
        wind_spd = map_flights['Wind_Spd'].iloc[0] if 'Wind_Spd' in map_flights.columns else '-'
        wind_dir = map_flights['Wind_Dir'].iloc[0] if 'Wind_Dir' in map_flights.columns else '-'
        
        c2.metric("기온 / 이슬점", f"{t_val}°C / {dew_val}°C" if pd.notna(t_val) else "-")
        c3.metric("풍속 / 풍향", f"{wind_spd}KT / {wind_dir}°" if pd.notna(wind_spd) else "-")
        c4.metric("기상 (WMO)", w_desc)

    if 'Lat' in map_flights.columns:
        m = folium.Map(location=[37.46, 126.44], zoom_start=13)
        runways = {'33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363), '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)}
        for r, c in runways.items(): folium.Marker(c, popup=r, icon=folium.Icon(color='gray', icon='plane')).add_to(m)

        color_dict = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange', 'Cancelled (CNL)': 'black'}
        for _, row in map_flights.iterrows():
            color = color_dict.get(row['Delay_Cause'], 'blue')
            popup = f"<b>{row['FLT']} ({row['STS_Detail']})</b><br>Delay: {row['Delay_Cause']}<br>Taxi: {row['Taxi_Time']:.1f}m<br>Total: {row['Total_Delay']:.0f}m"
            folium.Marker([row['Lat'], row['Lon']], popup=popup, tooltip=f"{row['FLT']}", icon=folium.Icon(color=color, icon='plane')).add_to(m)

        st_folium(m, width="100%", height=600)
    else:
        st.warning("지도 시각화를 위한 주기장 좌표 데이터가 부족합니다.")
