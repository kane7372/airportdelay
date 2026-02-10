import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import altair as alt
import os
import glob

st.set_page_config(page_title="Incheon Airport Ultimate Flight Analysis", layout="wide")

# ==========================================
# 1. 데이터 로드 (Flight + Weather)
# ==========================================
@st.cache_data
def load_data():
    # 1. Zone Data
    file_zone = 'rksi_stands_zoned.csv'
    if not os.path.exists(file_zone): return None, None, "Zone file not found"
    df_zone = pd.read_csv(file_zone)
    df_zone['Stand_ID'] = df_zone['Stand_ID'].astype(str)

    # 2. Flight Data (RAMP)
    ramp_files = glob.glob('*RAMP*.csv')
    if not ramp_files: return None, None, "No RAMP files found"
    
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
        try: return pd.to_datetime(f"20{date_str} {time_str}", format='%Y%m%d %H:%M')
        except: return pd.NaT

    # 1. STD (Schedule)
    df_flight['STD_Full'] = df_flight.apply(lambda x: parse_dt(x['Date'], x['STD']), axis=1)
    
    # 2. RAM & 3. ATD Calculation
    def calc_all_times(row):
        std = row['STD_Full']
        if pd.isna(std): return pd.NaT, pd.NaT, 0, 0
        
        # RAM Parsing
        try:
            ram_time = pd.to_datetime(row['RAM'], format='%H:%M').time()
            ram_dt = std.replace(hour=ram_time.hour, minute=ram_time.minute)
            if std.hour < 4 and ram_dt.hour > 20: ram_dt -= timedelta(days=1)
            elif std.hour > 20 and ram_dt.hour < 4: ram_dt += timedelta(days=1)
        except:
            ram_dt = pd.NaT

        # ATD Parsing
        atd_dt = pd.NaT
        try:
            if 'ATD' in row and pd.notna(row['ATD']):
                atd_time = pd.to_datetime(row['ATD'], format='%H:%M').time()
                # Base on RAM if available, else STD
                base_dt = ram_dt if not pd.isna(ram_dt) else std
                atd_dt = base_dt.replace(hour=atd_time.hour, minute=atd_time.minute)
                
                # If ATD is earlier than base (e.g. 00:10 vs 23:50), add 1 day
                if atd_dt < base_dt:
                    atd_dt += timedelta(days=1)
        except:
            pass

        # Metrics
        ramp_delay = 0
        taxi_time = 0
        
        if not pd.isna(ram_dt):
            ramp_delay = (ram_dt - std).total_seconds() / 60
            
        if not pd.isna(atd_dt) and not pd.isna(ram_dt):
            taxi_time = (atd_dt - ram_dt).total_seconds() / 60
        elif 'ATD-RAM' in row:
             try: taxi_time = float(row['ATD-RAM'])
             except: taxi_time = 0
             
        return ram_dt, atd_dt, ramp_delay, taxi_time

    res = df_flight.apply(calc_all_times, axis=1, result_type='expand')
    df_flight['RAM_Full'] = res[0]
    df_flight['ATD_Full'] = res[1]
    df_flight['Ramp_Delay'] = res[2]
    df_flight['Taxi_Time'] = res[3]
    
    def classify_delay(row):
        if row['Ramp_Delay'] < 15 and row['Taxi_Time'] < 25: return 'Normal'
        if row['Ramp_Delay'] >= 15 and row['Taxi_Time'] < 30: return 'Ramp (Gate)'
        elif row['Taxi_Time'] >= 30:
            if row['Ramp_Delay'] > (row['Taxi_Time'] - 20): return 'Ramp (Gate)'
            else: return 'Taxi (Ground)'
        else: return 'Ramp (Gate)'

    df_flight['Delay_Cause'] = df_flight.apply(classify_delay, axis=1)
    
    df_merged = pd.merge(df_flight, df_zone, left_on='SPT', right_on='Stand_ID', how='inner')

    # --- Weather Data ---
    if not df_weather.empty:
        df_weather['DT'] = pd.to_datetime(df_weather['일시'])
        df_weather = df_weather.rename(columns={
            '기온(°C)': 'Temp', '풍속(KT)': 'Wind_Spd', '풍향(deg)': 'Wind_Dir',
            '시정(m)': 'Visibility', '강수량(mm)': 'Precip'
        })
        df_weather['Precip'] = df_weather['Precip'].fillna(0)
        df_weather['Visibility'] = df_weather['Visibility'].fillna(10000)
        
    return df_merged, df_weather, "Success"

flights, weather, msg = load_data()

# ==========================================
# 2. UI & Interaction
# ==========================================
st.title("🛫 인천공항 종합 운항 분석 (STD / RAM / ATD)")

if flights is None:
    st.error(msg)
    st.stop()

# Sidebar
st.sidebar.header("설정 (Settings)")
min_dt, max_dt = flights['STD_Full'].min(), flights['STD_Full'].max()
sel_date = st.sidebar.date_input("날짜 선택", min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
sel_hour = st.sidebar.slider("시간대 선택", 0, 23, 12)

# Time Basis Selection
time_basis = st.sidebar.radio(
    "조회 기준 시간 (Time Basis)", 
    ["STD (계획)", "RAM (푸시백)", "ATD (실제 이륙)"],
    index=2 # Default to ATD
)

col_map = {"STD (계획)": "STD_Full", "RAM (푸시백)": "RAM_Full", "ATD (실제 이륙)": "ATD_Full"}
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

# ==========================================
# 3. Dashboard
# ==========================================
st.subheader(f"⏱️ 기준: {time_basis} | {sel_date} {sel_hour}:00")

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 대상 편수", f"{len(map_flights)} 편")
c2.metric("기온", f"{cur_weather['Temp']}°C" if cur_weather is not None else "-")
c3.metric("시정", f"{cur_weather['Visibility']:.0f}m" if cur_weather is not None else "-")
c4.metric("풍속", f"{cur_weather['Wind_Spd']}kt" if cur_weather is not None else "-")

# ==========================================
# 4. Charts & Map
# ==========================================
st.divider()
c_chart, c_map = st.columns([1, 2])

with c_chart:
    st.markdown("##### 📊 지연 원인 분포")
    if not map_flights.empty:
        chart_data = map_flights['Delay_Cause'].value_counts().reset_index()
        chart_data.columns = ['Cause', 'Count']
        
        base = alt.Chart(chart_data).encode(theta=alt.Theta("Count", stack=True))
        pie = base.mark_arc(outerRadius=120).encode(
            color=alt.Color("Cause", scale=alt.Scale(domain=['Normal', 'Ramp (Gate)', 'Taxi (Ground)'], range=['green', 'red', 'orange'])),
            order=alt.Order("Count", sort="descending"),
            tooltip=["Cause", "Count"]
        )
        st.altair_chart(pie, use_container_width=True)
    else:
        st.info("데이터 없음")

    st.markdown("##### ⚠️ 지연 Top 5")
    if not map_flights.empty:
        top5 = map_flights.sort_values('Ramp_Delay', ascending=False).head(5)
        st.dataframe(top5[['FLT', 'SPT', 'Ramp_Delay', 'Taxi_Time']], hide_index=True)

with c_map:
    st.markdown(f"##### 🗺️ {time_basis} 기준 주기장 현황")
    m = folium.Map(location=[37.46, 126.44], zoom_start=13)
    
    runways = {'33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363), '33R': (37.4563, 126.4647), '15L': (37.4838, 126.4402)}
    for r, c in runways.items(): folium.Marker(c, popup=r, icon=folium.Icon(color='gray', icon='plane')).add_to(m)

    color_dict = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange'}
    
    for _, row in map_flights.iterrows():
        color = color_dict.get(row['Delay_Cause'], 'blue')
        time_str = row[target_col].strftime('%H:%M')
        
        popup = f"<b>{row['FLT']}</b><br>{time_basis}: {time_str}<br>Delay: {row['Delay_Cause']}"
        folium.Marker(
            [row['Lat'], row['Lon']], popup=popup, tooltip=f"{row['FLT']} ({time_str})",
            icon=folium.Icon(color=color, icon='plane', prefix='fa')
        ).add_to(m)
        
    st_folium(m, width="100%", height=500)

# ==========================================
# 5. Weather Trend
# ==========================================
if weather is not None:
    st.divider()
    day_w = weather[weather['DT'].dt.date == sel_date].copy()
    if not day_w.empty:
        st.markdown("##### 📉 일간 기상 변화 (Temp vs Vis)")
        base = alt.Chart(day_w).encode(x=alt.X('DT:T', axis=alt.Axis(format='%H:%M')))
        line = base.mark_line(color='red').encode(y='Temp')
        area = base.mark_area(opacity=0.3, color='gray').encode(y='Visibility')
        st.altair_chart((line + area).resolve_scale(y='independent'), use_container_width=True)
