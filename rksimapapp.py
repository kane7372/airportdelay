import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import os
import glob

st.set_page_config(page_title="Incheon Airport Flight Monitor", layout="wide")

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
@st.cache_data
def load_and_process_data():
    file_zone = 'rksi_stands_zoned.csv'
    if not os.path.exists(file_zone):
        return None, "Zone file not found"

    ramp_files = glob.glob('*RAMP*.csv')
    if not ramp_files:
        return None, "No RAMP files found"
    
    df_list = []
    for f in ramp_files:
        try:
            d = pd.read_csv(f)
            df_list.append(d)
        except:
            pass
            
    if not df_list:
        return None, "Failed to read RAMP files"
        
    df_flight = pd.concat(df_list, ignore_index=True)
    df_zone = pd.read_csv(file_zone)
    
    df_flight['SPT'] = df_flight['SPT'].astype(str)
    df_zone['Stand_ID'] = df_zone['Stand_ID'].astype(str)
    df_flight['Date'] = df_flight['Date'].astype(str)
    
    def parse_dt(date_str, time_str):
        try:
            return pd.to_datetime(f"20{date_str} {time_str}", format='%Y%m%d %H:%M')
        except:
            return pd.NaT

    df_flight['STD_Full'] = df_flight.apply(lambda x: parse_dt(x['Date'], x['STD']), axis=1)
    
    def parse_ram(row):
        std = row['STD_Full']
        if pd.isna(std): return pd.NaT
        try:
            ram_time = pd.to_datetime(row['RAM'], format='%H:%M').time()
            ram_dt = std.replace(hour=ram_time.hour, minute=ram_time.minute)
            if std.hour < 4 and ram_dt.hour > 20:
                ram_dt -= timedelta(days=1)
            elif std.hour > 20 and ram_dt.hour < 4:
                ram_dt += timedelta(days=1)
            return ram_dt
        except:
            return pd.NaT

    df_flight['RAM_Full'] = df_flight.apply(parse_ram, axis=1)
    
    # STS 컬럼 결측치 처리 (기본값 DEP)
    if 'STS' not in df_flight.columns:
        df_flight['STS'] = 'DEP'
    df_flight['STS'] = df_flight['STS'].fillna('DEP')

    df_merged = pd.merge(df_flight, df_zone, left_on='SPT', right_on='Stand_ID', how='inner')
    
    return df_merged, f"Loaded {len(ramp_files)} files"

data, msg = load_and_process_data()

# ==========================================
# 2. UI 구성
# ==========================================
st.title("🛫 인천공항 주기장 운영 현황 (상태별 구분)")

if data is None:
    st.error(f"데이터 로드 실패: {msg}")
    st.stop()

st.sidebar.header("검색 조건")

min_dt = data['STD_Full'].min()
max_dt = data['STD_Full'].max()

if pd.isna(min_dt) or pd.isna(max_dt):
    st.error("날짜 데이터 오류")
    st.stop()

selected_date = st.sidebar.date_input("날짜 선택", min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
selected_hour = st.sidebar.slider("시간 선택", 0, 23, 12, format="%d:00")
time_mode = st.sidebar.radio("기준 시간", ["STD (계획)", "RAM (실제)"])
col_name = 'STD_Full' if "STD" in time_mode else 'RAM_Full'

# 데이터 필터링
filtered = data[
    (data[col_name].dt.date == selected_date) & 
    (data[col_name].dt.hour == selected_hour)
]

# ==========================================
# 3. 지도 시각화
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    m = folium.Map(location=[37.46, 126.44], zoom_start=13)
    
    # 활주로
    runways = {
        '33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363),
        '33R': (37.4563, 126.4647), '15L': (37.4838, 126.4402),
        '34L': (37.4411, 126.4377), '16R': (37.4680, 126.4130),
        '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)
    }
    for r, c in runways.items():
        folium.Marker(c, popup=r, icon=folium.Icon(color='gray', icon='plane')).add_to(m)

    # 빈 주기장
    all_spots = pd.read_csv('rksi_stands_zoned.csv')
    occupied_spots = filtered['SPT'].unique()
    empty_spots = all_spots[~all_spots['Stand_ID'].astype(str).isin(occupied_spots)]
    
    for _, row in empty_spots.iterrows():
        folium.CircleMarker(
            [row['Lat'], row['Lon']], radius=2, color='#DDDDDD', fill=True, fill_opacity=0.2,
            popup=f"Empty: {row['Stand_ID']}"
        ).add_to(m)

    # 상태별 색상 정의
    sts_colors = {
        'DEP': 'green',   # 정상 출발
        'DLA': 'orange',  # 지연 (Delay)
        'CNL': 'black',   # 결항 (Cancel)
        'DIV': 'blue'     # 회항 (Divert)
    }

    # 점유된 주기장
    for _, row in filtered.iterrows():
        sts = row.get('STS', 'DEP')
        color = sts_colors.get(sts, 'red') # 예외는 빨강
        
        popup_html = f"""
        <b>[{sts}] {row['FLT']}</b><br>
        Spot: {row['SPT']}<br>
        Dest: {row['DES']}<br>
        Time: {row[col_name].strftime('%H:%M')}
        """
        
        folium.Marker(
            [row['Lat'], row['Lon']],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"[{sts}] {row['FLT']}",
            icon=folium.Icon(color=color, icon='plane', prefix='fa')
        ).add_to(m)

    st_folium(m, width="100%", height=600)

with col2:
    st.subheader(f"📊 {selected_hour}시 현황")
    
    # 상태별 카운트 표시
    if not filtered.empty:
        counts = filtered['STS'].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("DEP (정상)", counts.get('DEP', 0))
        c2.metric("DLA (지연)", counts.get('DLA', 0))
        c3.metric("CNL (결항)", counts.get('CNL', 0))
    else:
        st.write("항공편 없음")
        
    st.divider()
    
    # 범례 (Legend)
    st.markdown("""
    **범례 (Legend):**
    - <span style='color:green'>●</span> **DEP**: 정상 출발 (Green)
    - <span style='color:orange'>●</span> **DLA**: 지연 (Orange)
    - <span style='color:black'>●</span> **CNL**: 결항 (Black)
    """, unsafe_allow_html=True)
    
    st.divider()
    st.write("📋 **Flight List**")
    if not filtered.empty:
        disp_cols = ['FLT', 'SPT', 'STS', 'DES', 'STD' if "STD" in time_mode else 'RAM']
        # 상태에 따라 색상 강조
        def highlight_sts(val):
            color = 'green' if val == 'DEP' else 'orange' if val == 'DLA' else 'black' if val == 'CNL' else 'blue'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(
            filtered[disp_cols].sort_values('SPT').style.applymap(highlight_sts, subset=['STS']),
            hide_index=True,
            use_container_width=True
        )
