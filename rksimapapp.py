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

st.set_page_config(page_title="Incheon Airport Ultimate Flight & Weather", layout="wide")

# ==========================================
# 1. Load Data
# ==========================================
@st.cache_data
def load_data():
    # 1. Zone
    file_zone = 'rksi_stands_zoned (2).csv'
    if not os.path.exists(file_zone):
        if os.path.exists('rksi_stands_zoned.csv'):
            file_zone = 'rksi_stands_zoned.csv'
        else:
            return None, None, None, None, "Zone file not found"
            
    df_zone = pd.read_csv(file_zone)
    df_zone['Stand_ID'] = df_zone['Stand_ID'].astype(str)

    # 2. Flight (RAMP, 도착편, 출발편 등 모두 로드)
    flight_files = glob.glob('*RAMP*.csv') + glob.glob('*출발*.csv') + glob.glob('*도착*.csv')
    if not flight_files: return None, None, None, None, "No Flight/RAMP files found"
    
    df_list = []
    for f in set(flight_files): # 중복 제거
        try:
            d = pd.read_csv(f)
            df_list.append(d)
        except: pass
    df_flight = pd.concat(df_list, ignore_index=True)
    
    # 3. Weather (AMOS)
    weather_files = glob.glob('AMOS_RKSI_*.csv') + glob.glob('기상_*.csv')
    df_weather = pd.DataFrame()
    if weather_files:
        w_list = []
        for f in set(weather_files):
            try:
                try: d = pd.read_csv(f, encoding='utf-8')
                except: d = pd.read_csv(f, encoding='cp949')
                w_list.append(d)
            except: pass
        if w_list:
            df_weather = pd.concat(w_list, ignore_index=True)

    # --- Preprocessing ---
    if 'SPT' in df_flight.columns:
        df_flight['SPT'] = df_flight['SPT'].astype(str)
    df_flight['Date_str'] = df_flight['Date'].astype(str)
    
    def parse_dt(date_str, time_str):
        try: 
            d_str = str(date_str).strip()
            t_str = str(time_str).strip()
            if len(d_str) == 8: return pd.to_datetime(f"{d_str} {t_str}", format='%Y%m%d %H:%M')
            else: return pd.to_datetime(f"20{d_str} {t_str}", format='%Y%m%d %H:%M')
        except: return pd.NaT

    df_flight['Plan_Time'] = df_flight['STD'] if 'STD' in df_flight.columns else df_flight.get('STA', pd.NaT)
    df_flight['STD_Full'] = df_flight.apply(lambda x: parse_dt(x['Date_str'], x['Plan_Time']), axis=1)
    
    def calc_all_times(row):
        std = row['STD_Full']
        if pd.isna(std): return pd.NaT, pd.NaT, 0, 0, 0
        
        # RAM Parsing
        ram_dt = pd.NaT
        try:
            if pd.notna(row.get('RAM')):
                ram_time = pd.to_datetime(str(row['RAM']), format='%H:%M').time()
                ram_dt = std.replace(hour=ram_time.hour, minute=ram_time.minute)
                if std.hour < 4 and ram_dt.hour > 20: ram_dt -= timedelta(days=1)
                elif std.hour > 20 and ram_dt.hour < 4: ram_dt += timedelta(days=1)
        except: pass

        # ATD/ATA Parsing
        act_dt = pd.NaT
        act_col = 'ATD' if 'ATD' in row and pd.notna(row['ATD']) else ('ATA' if 'ATA' in row and pd.notna(row['ATA']) else None)
        try:
            if act_col:
                act_time = pd.to_datetime(str(row[act_col]), format='%H:%M').time()
                base_dt = ram_dt if not pd.isna(ram_dt) else std
                act_dt = base_dt.replace(hour=act_time.hour, minute=act_time.minute)
                if act_dt < base_dt: act_dt += timedelta(days=1)
        except: pass

        ramp_delay = 0
        taxi_time = 0
        total_delay = 0
        
        if not pd.isna(act_dt): total_delay = abs((act_dt - std).total_seconds() / 60)
        if not pd.isna(ram_dt): ramp_delay = abs((ram_dt - std).total_seconds() / 60)
            
        if 'ATD-RAM' in row and pd.notna(row['ATD-RAM']):
             try: taxi_time = float(row['ATD-RAM'])
             except: taxi_time = 0
        elif 'RAM-ATA' in row and pd.notna(row['RAM-ATA']):
             try: taxi_time = float(row['RAM-ATA'])
             except: taxi_time = 0
        elif not pd.isna(act_dt) and not pd.isna(ram_dt):
            taxi_time = abs((act_dt - ram_dt).total_seconds() / 60)
             
        return ram_dt, act_dt, ramp_delay, taxi_time, total_delay

    res = df_flight.apply(calc_all_times, axis=1, result_type='expand')
    df_flight['RAM_Full'] = res[0]
    df_flight['ATD_Full'] = res[1]
    df_flight['Ramp_Delay'] = res[2]
    df_flight['Taxi_Time'] = res[3]
    df_flight['Total_Delay'] = res[4]
    
    # --- Statistics ---
    df_flight['YM'] = df_flight['STD_Full'].dt.to_period('M')
    valid_taxi = df_flight[df_flight['Taxi_Time'] > 0]
    stats = valid_taxi.groupby('YM')['Taxi_Time'].agg(['mean', 'std']).reset_index()
    stats['Limit_1Sigma'] = stats['mean'] + stats['std']
    
    df_flight = pd.merge(df_flight, stats, on='YM', how='left')
    
    def classify_delay_stat(row):
        if row['Total_Delay'] <= 15: return 'Normal'
        limit = row['Limit_1Sigma'] if pd.notna(row['Limit_1Sigma']) else 30
        if row['Taxi_Time'] >= limit: return 'Taxi (Ground)'
        else: return 'Ramp (Gate)'

    df_flight['Delay_Cause'] = df_flight.apply(classify_delay_stat, axis=1)
    df_merged = pd.merge(df_flight, df_zone, left_on='SPT', right_on='Stand_ID', how='inner') if 'SPT' in df_flight.columns else df_flight

    # --- WMO 4677 Weather Mapping ---
    if not df_weather.empty:
        df_weather['DT'] = pd.to_datetime(df_weather['일시'])
        df_weather = df_weather.rename(columns={
            '기온(°C)': 'Temp', '풍속(KT)': 'Wind_Spd', '풍향(deg)': 'Wind_Dir',
            '시정(m)': 'Visibility', '강수량(mm)': 'Precip', '일기현상': 'W_Code'
        })
        df_weather['Precip'] = df_weather['Precip'].fillna(0)
        df_weather['Visibility'] = df_weather['Visibility'].fillna(10000)
        
        # WMO 4677 기반 기상 코드 분류
        def parse_wmo_code(code):
            if pd.isna(code): return "기상현상 없음"
            try: c = int(code)
            except: return "기타"
            
            if c < 10: return "맑음/흐림/연무"
            elif 10 <= c <= 19: return "박무/안개(부분)"
            elif 20 <= c <= 29: 
                if c in [22, 26]: return "눈(최근)"
                elif c == 23: return "진눈깨비(최근)"
                elif c == 24: return "어는 비(최근)"
                else: return "비/안개(최근)"
            elif 30 <= c <= 35: return "황사/모래폭풍"
            elif 36 <= c <= 39: return "날림눈/눈보라"
            elif 40 <= c <= 49: return "안개/빙무"
            elif 50 <= c <= 55: return "안개비"
            elif 56 <= c <= 57: return "어는 안개비 (결빙)"
            elif 58 <= c <= 59: return "비 섞인 안개비"
            elif 60 <= c <= 65: return "비"
            elif 66 <= c <= 67: return "어는 비 (결빙)"
            elif 68 <= c <= 69: return "진눈깨비"
            elif 70 <= c <= 79: return "강설 (눈)"
            elif 80 <= c <= 82: return "소나기"
            elif 83 <= c <= 84: return "진눈깨비 소나기"
            elif 85 <= c <= 86: return "눈 소나기"
            elif 87 <= c <= 90: return "우박/싸락눈 소나기"
            elif 91 <= c <= 99:
                if c in [93, 94]: return "눈 동반 뇌전"
                else: return "비/우박 동반 뇌전"
            else: return "기타"
            
        df_weather['Weather_Desc'] = df_weather['W_Code'].apply(parse_wmo_code)
        
        # 강설/결빙 상태 판단 (제빙이 필요한 악기상 키워드 필터링)
        def is_winter_hazard(desc):
            hazards = ['눈', '설', '진눈깨비', '어는', '빙무', '우박']
            return any(k in desc for k in hazards)

        daily_weather = df_weather.groupby(df_weather['DT'].dt.date)['Weather_Desc'].apply(
            lambda x: '강설/결빙 (De-icing)' if any(is_winter_hazard(w) for w in x) else '일반'
        ).reset_index()
        daily_weather.columns = ['Date_Only', 'Snow_Status']
        
        df_merged['Date_Only'] = df_merged['STD_Full'].dt.date
        df_merged = pd.merge(df_merged, daily_weather, on='Date_Only', how='left')
        df_merged['Snow_Status'] = df_merged['Snow_Status'].fillna('일반')
        
    return df_merged, df_weather, stats, df_zone, "Success"

flights, weather, taxi_stats, zone_data, msg = load_data()

if flights is None:
    st.error(msg)
    st.stop()

# ==========================================
# 2. UI Layout
# ==========================================
st.title("🛫 인천공항 겨울철 지연 분석 대시보드 (WMO 기반)")

tab1, tab2 = st.tabs(["📊 거시적 트렌드 (장기 분석)", "🔎 상세 지연 및 지도 (일간/시간별)"])

# ==========================================
# [TAB 1] 장기 트렌드 분석
# ==========================================
with tab1:
    st.header("1. 전체 운항편수 및 기상 조건별 비교")
    if 'STS' in flights.columns:
        trend_df = flights.groupby(['YM', 'STS', 'Snow_Status']).size().reset_index(name='Count')
        trend_df['YM'] = trend_df['YM'].astype(str)
        fig_trend = px.bar(
            trend_df, x='YM', y='Count', color='STS', facet_col='Snow_Status',
            title="월별 출/도착 운항편 수", barmode='group'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.divider()
    st.header("2. 일별 평균 지상이동시간 (Taxi Time) 추이")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("🛫 출발편 (ATD-RAM)")
        dep_flights = flights[flights['STS'] == 'DEP'] if 'STS' in flights.columns else flights
        if not dep_flights.empty:
            daily_taxi_dep = dep_flights.groupby(['Date_Only', 'Snow_Status'])['Taxi_Time'].mean().reset_index()
            # 색상 매핑 명확화: 강설/결빙은 빨간색 계열
            fig_dep = px.line(daily_taxi_dep, x='Date_Only', y='Taxi_Time', color='Snow_Status', 
                              color_discrete_map={'강설/결빙 (De-icing)': 'red', '일반': 'blue'}, markers=True)
            st.plotly_chart(fig_dep, use_container_width=True)
            
    with col_t2:
        st.subheader("🛬 도착편 (RAM-ATA)")
        arr_flights = flights[flights['STS'] == 'ARR'] if 'STS' in flights.columns else pd.DataFrame()
        if not arr_flights.empty:
            daily_taxi_arr = arr_flights.groupby(['Date_Only', 'Snow_Status'])['Taxi_Time'].mean().reset_index()
            fig_arr = px.line(daily_taxi_arr, x='Date_Only', y='Taxi_Time', color='Snow_Status', 
                              color_discrete_map={'강설/결빙 (De-icing)': 'red', '일반': 'blue'}, markers=True)
            st.plotly_chart(fig_arr, use_container_width=True)

# ==========================================
# [TAB 2] 일간/시간별 상세 분석
# ==========================================
with tab2:
    st.sidebar.header("설정 (Settings)")
    min_dt, max_dt = flights['STD_Full'].dropna().min(), flights['STD_Full'].dropna().max()
    sel_date = st.sidebar.date_input("날짜 선택", min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
    sel_hour = st.sidebar.slider("시간대 선택", 0, 23, 12)

    time_basis = st.sidebar.radio("기준 시간", ["STD (계획)", "RAM (푸시백)"], index=1)
    col_map = {"STD (계획)": "STD_Full", "RAM (푸시백)": "RAM_Full"}
    target_col = col_map[time_basis]

    valid_flights = flights.dropna(subset=[target_col]).copy()
    day_flights = valid_flights[valid_flights[target_col].dt.date == sel_date].copy()
    map_flights = day_flights[day_flights[target_col].dt.hour == sel_hour].copy()

    cur_weather = None
    if weather is not None and not weather.empty:
        target_dt = pd.to_datetime(f"{sel_date} {sel_hour}:00")
        w_row = weather[weather['DT'] == target_dt]
        if not w_row.empty: cur_weather = w_row.iloc[0]

    def get_cardinal(deg):
        if pd.isna(deg): return ""
        dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = int((deg + 22.5) / 45.0) % 8
        return dirs[idx]

    # --- Header ---
    st.subheader(f"⏱️ {time_basis} 기준 | {sel_date} {sel_hour}:00")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("대상 편수", f"{len(map_flights)}")
    c2.metric("기온", f"{cur_weather['Temp']}°C" if cur_weather is not None else "-")
    c3.metric("시정", f"{cur_weather['Visibility']:.0f}m" if cur_weather is not None else "-")

    wd, ws = cur_weather['Wind_Dir'] if cur_weather is not None else None, cur_weather['Wind_Spd'] if cur_weather is not None else None
    wind_str = f"{ws}kt ({wd:.0f}° {get_cardinal(wd)})" if wd is not None else "-"
    c4.metric("풍속/풍향", wind_str)
    
    # WMO 기반 상세 기상 코드가 여기에 표시됩니다!
    w_desc = cur_weather['Weather_Desc'] if cur_weather is not None else "-"
    c5.metric("기상 현상 (WMO)", w_desc)
    c6.metric("강수량", f"{cur_weather['Precip']}mm" if cur_weather is not None else "-")

    # --- Statistics Table ---
    st.divider()
    st.markdown("##### 📊 월별 Taxi Time 통계 기준표 (Sigma Analysis)")
    current_limit = 30.0 

    if taxi_stats is not None:
        disp_stats = taxi_stats.copy()
        disp_stats['YM'] = disp_stats['YM'].astype(str)
        disp_stats = disp_stats.rename(columns={'YM': '연월', 'mean': '평균 (분)', 'std': '표준편차', 'Limit_1Sigma': '1σ (주의)'})
        st.dataframe(disp_stats[['연월', '평균 (분)', '표준편차', '1σ (주의)']].style.format({'평균 (분)': '{:.1f}', '표준편차': '{:.1f}', '1σ (주의)': '{:.1f}'}), hide_index=True, use_container_width=True)
        
        curr_ym = pd.Period(sel_date, freq='M')
        curr_stat = taxi_stats[taxi_stats['YM'] == curr_ym]
        
        if not curr_stat.empty:
            current_limit = curr_stat.iloc[0]['Limit_1Sigma']
            st.info(f"💡 **{curr_ym}월 지연 기준:** Total Delay ≥ 15분")
            st.write(f"   - **Taxi (Ground):** Taxi Time ≥ {current_limit:.1f}분 (1σ)")
            st.write(f"   - **Ramp (Gate):** Taxi Time < {current_limit:.1f}분")

    # --- Scatter Plot ---
    st.divider()
    st.markdown("##### 📈 지연 원인 분석 (Total Delay ≥ 15분 기준)")
    col_chart, col_dummy = st.columns([3, 1])
    with col_chart:
        if not day_flights.empty:
            scatter = alt.Chart(day_flights).mark_circle(size=60).encode(
                x=alt.X('Ramp_Delay', title='주기장 지연 (분)'),
                y=alt.Y('Taxi_Time', title='지상 이동 시간 (분)'),
                color=alt.Color('Delay_Cause', 
                                scale=alt.Scale(domain=['Normal', 'Ramp (Gate)', 'Taxi (Ground)'], range=['green', 'red', 'orange']),
                                legend=alt.Legend(title="지연 원인")),
                tooltip=['FLT', 'SPT', 'Delay_Cause', 'Total_Delay', 'Ramp_Delay', 'Taxi_Time']
            ).interactive()
            
            rule_taxi = alt.Chart(pd.DataFrame({'y': [current_limit]})).mark_rule(color='orange', strokeDash=[3,3]).encode(y='y')
            st.altair_chart(scatter + rule_taxi, use_container_width=True)

    with col_dummy:
        st.info("💡 **판정 로직**")
        st.write("1. **Normal:** 이륙 지연 15분 미만")
        st.write("2. **Delayed:** 15분 이상 시 원인 분류")
        st.write(f"   - <span style='color:orange'>●</span> **Taxi (Ground):** Taxi Time ≥ {current_limit:.1f}분", unsafe_allow_html=True)
        st.write("   - <span style='color:red'>●</span> **Ramp (Gate):** Taxi Time < 1σ", unsafe_allow_html=True)

    # --- Map Visualization ---
    st.divider()
    st.markdown(f"##### 🗺️ {time_basis} 기준 주기장 현황")
    if 'Lat' in map_flights.columns:
        m = folium.Map(location=[37.46, 126.44], zoom_start=13)
        runways = {
            '33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363),
            '33R': (37.4563, 126.4647), '15L': (37.4838, 126.4402),
            '34L': (37.4411, 126.4377), '16R': (37.4680, 126.4130),
            '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)
        }
        for r, c in runways.items(): folium.Marker(c, popup=r, icon=folium.Icon(color='gray', icon='plane')).add_to(m)

        deicing_spots = zone_data[zone_data['Category'] == 'De-icing Apron']
        for _, row in deicing_spots.iterrows():
            folium.CircleMarker(
                [row['Lat'], row['Lon']], radius=3, color='cyan', fill=True, fill_opacity=0.6,
                popup=f"De-icing Spot: {row['Stand_ID']}", tooltip=f"De-icing {row['Stand_ID']}"
            ).add_to(m)

        color_dict = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange'}
        for _, row in map_flights.iterrows():
            color = color_dict.get(row['Delay_Cause'], 'blue')
            popup = f"<b>{row['FLT']}</b><br>Delay: {row['Delay_Cause']}<br>Total: {row['Total_Delay']:.0f}m"
            folium.Marker(
                [row['Lat'], row['Lon']], popup=popup, tooltip=f"{row['FLT']}",
                icon=folium.Icon(color=color, icon='plane', prefix='fa')
            ).add_to(m)

        st_folium(m, width="100%", height=700)
    else:
        st.warning("지도 시각화를 위한 좌표 데이터가 부족합니다.")

    # --- Weather Trend ---
    if weather is not None:
        st.divider()
        day_w = weather[weather['DT'].dt.date == sel_date].copy()
        if not day_w.empty:
            st.markdown("##### 📉 일간 기상 변화")
            base = alt.Chart(day_w).encode(x=alt.X('DT:T', axis=alt.Axis(format='%H:%M', title='시간')))
            line = base.mark_line(color='red').encode(y=alt.Y('Temp', title='기온'))
            area = base.mark_area(opacity=0.3, color='gray').encode(y=alt.Y('Visibility', title='시정'))
            
            # WMO 기상 코드 40 이상 (안개, 비, 눈 등 악기상) 필터링하여 포인트 표시
            bad_weather = day_w[day_w['W_Code'].notna() & (day_w['W_Code'] >= 40)].copy()
            points = alt.Chart(bad_weather).mark_point(color='blue', size=100, shape='triangle-up').encode(
                x='DT:T', y='Temp', tooltip=['DT', 'Weather_Desc']
            )
            st.altair_chart((line + area + points).resolve_scale(y='independent'), use_container_width=True)
