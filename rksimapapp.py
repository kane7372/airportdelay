import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import altair as alt
import os
import glob
import numpy as np

st.set_page_config(page_title="Incheon Airport Statistical Analysis Final", layout="wide")

# ==========================================
# 1. Load Data (Flight + Weather)
# ==========================================
@st.cache_data
def load_data():
    # 1. Zone Data
    file_zone = 'rksi_stands_zoned.csv'
    if not os.path.exists(file_zone): return None, None, None, "Zone file not found"
    df_zone = pd.read_csv(file_zone)
    df_zone['Stand_ID'] = df_zone['Stand_ID'].astype(str)

    # 2. Flight Data (RAMP)
    ramp_files = glob.glob('*RAMP*.csv')
    if not ramp_files: return None, None, None, "No RAMP files found"
    
    df_list = []
    for f in ramp_files:
        try:
            d = pd.read_csv(f)
            df_list.append(d)
        except: pass
    df_flight = pd.concat(df_list, ignore_index=True)
    
    # 3. Weather Data (AMOS)
    weather_files = glob.glob('AMOS_RKSI_*.csv')
    df_weather = pd.DataFrame()
    if weather_files:
        w_list = []
        for f in weather_files:
            try:
                try: d = pd.read_csv(f, encoding='utf-8')
                except: d = pd.read_csv(f, encoding='cp949')
                w_list.append(d)
            except: pass
        if w_list:
            df_weather = pd.concat(w_list, ignore_index=True)

    # --- Preprocessing Flight Data ---
    df_flight['SPT'] = df_flight['SPT'].astype(str)
    df_flight['Date'] = df_flight['Date'].astype(str)
    
    def parse_dt(date_str, time_str):
        try: 
            # Force string type to avoid formatting errors
            d_str = str(date_str).strip()
            t_str = str(time_str).strip()
            return pd.to_datetime(f"20{d_str} {t_str}", format='%Y%m%d %H:%M')
        except: return pd.NaT

    df_flight['STD_Full'] = df_flight.apply(lambda x: parse_dt(x['Date'], x['STD']), axis=1)
    
    # Calculate Times & Delays
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

        # ATD Parsing
        atd_dt = pd.NaT
        try:
            if 'ATD' in row and pd.notna(row['ATD']):
                atd_time = pd.to_datetime(str(row['ATD']), format='%H:%M').time()
                base_dt = ram_dt if not pd.isna(ram_dt) else std
                atd_dt = base_dt.replace(hour=atd_time.hour, minute=atd_time.minute)
                if atd_dt < base_dt: atd_dt += timedelta(days=1)
        except: pass

        # Metrics Calculation
        ramp_delay = 0
        taxi_time = 0
        total_delay = 0
        
        # Total Delay: ATD - STD
        if not pd.isna(atd_dt):
            total_delay = (atd_dt - std).total_seconds() / 60
        
        # Ramp Delay
        if not pd.isna(ram_dt):
            ramp_delay = (ram_dt - std).total_seconds() / 60
            
        # Taxi Time
        if 'ATD-RAM' in row and pd.notna(row['ATD-RAM']):
             try: taxi_time = float(row['ATD-RAM'])
             except: taxi_time = 0
        elif not pd.isna(atd_dt) and not pd.isna(ram_dt):
            taxi_time = (atd_dt - ram_dt).total_seconds() / 60
             
        return ram_dt, atd_dt, ramp_delay, taxi_time, total_delay

    res = df_flight.apply(calc_all_times, axis=1, result_type='expand')
    df_flight['RAM_Full'] = res[0]
    df_flight['ATD_Full'] = res[1]
    df_flight['Ramp_Delay'] = res[2]
    df_flight['Taxi_Time'] = res[3]
    df_flight['Total_Delay'] = res[4]
    
    # --- Statistical Thresholding ---
    df_flight['YM'] = df_flight['STD_Full'].dt.to_period('M')
    
    valid_taxi = df_flight[df_flight['Taxi_Time'] > 0]
    stats = valid_taxi.groupby('YM')['Taxi_Time'].agg(['mean', 'std']).reset_index()
    stats['Limit_1Sigma'] = stats['mean'] + stats['std']
    
    df_flight = pd.merge(df_flight, stats, on='YM', how='left')
    
    def classify_delay_stat(row):
        # 1. Total Delay Condition (Primary)
        if row['Total_Delay'] <= 15:
            return 'Normal'
            
        # 2. If Delayed, Determine Cause (Secondary)
        if row['Ramp_Delay'] >= 15:
            return 'Ramp (Gate)'
            
        # Check Sigma Limit for Taxi
        limit = row['Limit_1Sigma'] if pd.notna(row['Limit_1Sigma']) else 30
        if row['Taxi_Time'] > limit:
            return 'Taxi (Ground)'
            
        return 'Taxi (Ground)'

    df_flight['Delay_Cause'] = df_flight.apply(classify_delay_stat, axis=1)
    df_merged = pd.merge(df_flight, df_zone, left_on='SPT', right_on='Stand_ID', how='inner')

    # --- Weather Data ---
    if not df_weather.empty:
        df_weather['DT'] = pd.to_datetime(df_weather['일시'])
        df_weather = df_weather.rename(columns={
            '기온(°C)': 'Temp', '풍속(KT)': 'Wind_Spd', '풍향(deg)': 'Wind_Dir',
            '시정(m)': 'Visibility', '강수량(mm)': 'Precip', '일기현상': 'W_Code'
        })
        df_weather['Precip'] = df_weather['Precip'].fillna(0)
        df_weather['Visibility'] = df_weather['Visibility'].fillna(10000)
        
        def parse_weather_code(code):
            if pd.isna(code): return "맑음/박무"
            try: c = int(code)
            except: return "기타"
            if 40 <= c <= 49: return "안개/빙무"
            elif 50 <= c <= 59: return "안개비"
            elif 60 <= c <= 67: return "비"
            elif 68 <= c <= 69: return "진눈깨비"
            elif 70 <= c <= 79: return "강설"
            elif 80 <= c <= 99: return "소낙성/뇌전"
            else: return "기타"
        df_weather['Weather_Desc'] = df_weather['W_Code'].apply(parse_weather_code)
        
    return df_merged, df_weather, stats, "Success"

flights, weather, taxi_stats, msg = load_data()

# ==========================================
# 2. UI & Interaction
# ==========================================
st.title("🛫 인천공항 통계 기반 지연 분석 (Final Fixed)")

if flights is None:
    st.error(msg)
    st.stop()

# Sidebar
st.sidebar.header("설정 (Settings)")
min_dt, max_dt = flights['STD_Full'].min(), flights['STD_Full'].max()
sel_date = st.sidebar.date_input("날짜 선택", min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
sel_hour = st.sidebar.slider("시간대 선택", 0, 23, 12)

time_basis = st.sidebar.radio("기준 시간", ["STD (계획)", "RAM (푸시백)"], index=1)
col_map = {"STD (계획)": "STD_Full", "RAM (푸시백)": "RAM_Full"}
target_col = col_map[time_basis]

# Filter Data
valid_flights = flights.dropna(subset=[target_col]).copy()
day_flights = valid_flights[valid_flights[target_col].dt.date == sel_date].copy()
map_flights = day_flights[day_flights[target_col].dt.hour == sel_hour].copy()

# Weather
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

# ==========================================
# 3. Dashboard Header
# ==========================================
st.subheader(f"⏱️ {time_basis} 기준 | {sel_date} {sel_hour}:00")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("대상 편수", f"{len(map_flights)}")
c2.metric("기온", f"{cur_weather['Temp']}°C" if cur_weather is not None else "-")
c3.metric("시정", f"{cur_weather['Visibility']:.0f}m" if cur_weather is not None else "-")

wd, ws = cur_weather['Wind_Dir'] if cur_weather is not None else None, cur_weather['Wind_Spd'] if cur_weather is not None else None
wind_str = f"{ws}kt ({wd:.0f}° {get_cardinal(wd)})" if wd is not None else "-"
c4.metric("풍속/풍향", wind_str)

w_desc = cur_weather['Weather_Desc'] if cur_weather is not None else "-"
c5.metric("기상 현상", w_desc)
c6.metric("강수량", f"{cur_weather['Precip']}mm" if cur_weather is not None else "-")

# ==========================================
# 4. Statistical Analysis Table
# ==========================================
st.divider()
st.markdown("##### 📊 월별 Taxi Time 통계 기준표 (Sigma Analysis)")

current_limit = 30.0 

if taxi_stats is not None:
    disp_stats = taxi_stats.copy()
    disp_stats['YM'] = disp_stats['YM'].astype(str)
    disp_stats = disp_stats.rename(columns={
        'YM': '연월', 'mean': '평균 (분)', 'std': '표준편차', 'Limit_1Sigma': '1σ (주의)'
    })
    st.dataframe(disp_stats[['연월', '평균 (분)', '표준편차', '1σ (주의)']].style.format('{:.1f}'), hide_index=True, use_container_width=True)
    
    curr_ym = pd.Period(sel_date, freq='M')
    curr_stat = taxi_stats[taxi_stats['YM'] == curr_ym]
    
    if not curr_stat.empty:
        current_limit = curr_stat.iloc[0]['Limit_1Sigma']
        st.info(f"💡 **{curr_ym}월 지연 기준:** Total Delay > 15분 AND (Ramp Delay > 15분 OR Taxi Time > {current_limit:.1f}분)")
    else:
        st.warning(f"⚠️ **{curr_ym}월 통계 없음:** 기본값 30분 기준 사용")

# ==========================================
# 5. Scatter Plot (Analysis)
# ==========================================
st.divider()
st.markdown("##### 📈 지연 원인 분석 (Total Delay > 15분 기준)")

col_chart, col_dummy = st.columns([3, 1])
with col_chart:
    if not day_flights.empty:
        scatter = alt.Chart(day_flights).mark_circle(size=60).encode(
            x=alt.X('Ramp_Delay', title='주기장 지연 (분)'),
            y=alt.Y('Taxi_Time', title='지상 이동 시간 (분)'),
            color=alt.Color('Delay_Cause', 
                            scale=alt.Scale(domain=['Normal', 'Ramp (Gate)', 'Taxi (Ground)'],
                                            range=['green', 'red', 'orange']),
                            legend=alt.Legend(title="지연 원인")),
            tooltip=['FLT', 'SPT', 'Delay_Cause', 'Total_Delay', 'Ramp_Delay', 'Taxi_Time']
        ).interactive()
        
        rule_taxi = alt.Chart(pd.DataFrame({'y': [current_limit]})).mark_rule(color='orange', strokeDash=[3,3]).encode(y='y')
        rule_ramp = alt.Chart(pd.DataFrame({'x': [15]})).mark_rule(color='red', strokeDash=[3,3]).encode(x='x')
        
        st.altair_chart(scatter + rule_taxi + rule_ramp, use_container_width=True)
    else:
        st.info("데이터 없음")

with col_dummy:
    st.info("💡 **판정 로직**")
    st.write("1. **Normal (Green):** 이륙 지연(ATD-STD) 15분 이하")
    st.write("2. **Delayed:** 15분 초과 시 원인 분류")
    st.write("   - <span style='color:red'>●</span> **Ramp (Gate):** 주기장 지연 ≥ 15분", unsafe_allow_html=True)
    st.write(f"   - <span style='color:orange'>●</span> **Taxi (Ground):** Taxi Time > {current_limit:.1f}분 (1σ)", unsafe_allow_html=True)

# ==========================================
# 6. Map Visualization
# ==========================================
st.divider()
st.markdown(f"##### 🗺️ {time_basis} 기준 주기장 현황")

m = folium.Map(location=[37.46, 126.44], zoom_start=13)
runways = {
    '33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363),
    '33R': (37.4563, 126.4647), '15L': (37.4838, 126.4402),
    '34L': (37.4411, 126.4377), '16R': (37.4680, 126.4130),
    '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)
}
for r, c in runways.items(): folium.Marker(c, popup=r, icon=folium.Icon(color='gray', icon='plane')).add_to(m)

color_dict = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange'}

for _, row in map_flights.iterrows():
    color = color_dict.get(row['Delay_Cause'], 'blue')
    time_str = row[target_col].strftime('%H:%M')
    
    popup = f"<b>{row['FLT']}</b><br>Delay: {row['Delay_Cause']}<br>Total: {row['Total_Delay']:.0f}m"
    folium.Marker(
        [row['Lat'], row['Lon']], popup=popup, tooltip=f"{row['FLT']}",
        icon=folium.Icon(color=color, icon='plane', prefix='fa')
    ).add_to(m)

st_folium(m, width="100%", height=700)

# ==========================================
# 7. Weather Trend
# ==========================================
if weather is not None:
    st.divider()
    day_w = weather[weather['DT'].dt.date == sel_date].copy()
    if not day_w.empty:
        st.markdown("##### 📉 일간 기상 변화")
        base = alt.Chart(day_w).encode(x=alt.X('DT:T', axis=alt.Axis(format='%H:%M', title='시간')))
        line = base.mark_line(color='red').encode(y=alt.Y('Temp', title='기온'))
        area = base.mark_area(opacity=0.3, color='gray').encode(y=alt.Y('Visibility', title='시정'))
        
        bad_weather = day_w[day_w['W_Code'].notna() & (day_w['W_Code'] >= 40)].copy()
        points = alt.Chart(bad_weather).mark_point(color='blue', size=100, shape='triangle-up').encode(
            x='DT:T', y='Temp', tooltip=['DT', 'Weather_Desc']
        )
        st.altair_chart((line + area + points).resolve_scale(y='independent'), use_container_width=True)
