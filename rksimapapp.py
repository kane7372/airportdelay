import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import os
import glob

st.set_page_config(page_title="Incheon Airport 3-Year Flight Monitor", layout="wide")

# ==========================================
# 1. 데이터 로드 및 전처리 (Caching)
# ==========================================
@st.cache_data
def load_and_process_data():
    # 1. Zone 파일 확인
    file_zone = 'rksi_stands_zoned.csv'
    if not os.path.exists(file_zone):
        return None, "Zone file not found"

    # 2. RAMP 파일들 자동 검색 (이름에 'RAMP'가 포함된 모든 CSV)
    ramp_files = glob.glob('*RAMP*.csv')
    if not ramp_files:
        return None, "No RAMP files found"
    
    # 3. 데이터 로드 및 병합
    df_list = []
    for f in ramp_files:
        try:
            d = pd.read_csv(f)
            # 파일별로 컬럼명이 조금 다를 수 있으니 공통 컬럼 위주로 처리
            # 필수 컬럼: Date, STD, RAM, SPT, FLT
            df_list.append(d)
        except:
            pass
            
    if not df_list:
        return None, "Failed to read RAMP files"
        
    df_flight = pd.concat(df_list, ignore_index=True)
    df_zone = pd.read_csv(file_zone)
    
    # 데이터 타입 통일
    df_flight['SPT'] = df_flight['SPT'].astype(str)
    df_zone['Stand_ID'] = df_zone['Stand_ID'].astype(str)
    df_flight['Date'] = df_flight['Date'].astype(str)
    
    # 날짜/시간 파싱 함수 (YYMMDD -> YYYY-MM-DD)
    def parse_dt(date_str, time_str):
        try:
            # 6자리 날짜(230101) + 4자리 시간(12:30)
            return pd.to_datetime(f"20{date_str} {time_str}", format='%Y%m%d %H:%M')
        except:
            return pd.NaT

    # STD (스케줄) 기준 시간 생성
    df_flight['STD_Full'] = df_flight.apply(lambda x: parse_dt(x['Date'], x['STD']), axis=1)
    
    # RAM (실제) 기준 시간 생성 (날짜 변경선 처리)
    def parse_ram(row):
        std = row['STD_Full']
        if pd.isna(std): return pd.NaT
        try:
            ram_time = pd.to_datetime(row['RAM'], format='%H:%M').time()
            ram_dt = std.replace(hour=ram_time.hour, minute=ram_time.minute)
            
            # STD가 00~03시인데 RAM이 20~23시면 -> 전날로 간주
            if std.hour < 4 and ram_dt.hour > 20:
                ram_dt -= timedelta(days=1)
            # STD가 20~23시인데 RAM이 00~03시면 -> 다음날로 간주
            elif std.hour > 20 and ram_dt.hour < 4:
                ram_dt += timedelta(days=1)
            return ram_dt
        except:
            return pd.NaT

    df_flight['RAM_Full'] = df_flight.apply(parse_ram, axis=1)
    
    # 좌표 병합 (Inner Join)
    df_merged = pd.merge(df_flight, df_zone, left_on='SPT', right_on='Stand_ID', how='inner')
    
    return df_merged, f"Loaded {len(ramp_files)} files"

# 데이터 로딩 실행
data, msg = load_and_process_data()

# ==========================================
# 2. UI 구성
# ==========================================
st.title("🛫 인천공항 3개년 주기장 운영 현황")

if data is None:
    st.error(f"데이터 로드 실패: {msg}")
    st.stop()

# 사이드바 설정
st.sidebar.header("검색 조건")

# 날짜 범위 확인 및 선택
min_dt = data['STD_Full'].min()
max_dt = data['STD_Full'].max()

if pd.isna(min_dt) or pd.isna(max_dt):
    st.error("날짜 데이터가 유효하지 않습니다.")
    st.stop()

selected_date = st.sidebar.date_input(
    "날짜 선택 (Date)", 
    min_dt.date(), 
    min_value=min_dt.date(), 
    max_value=max_dt.date()
)

selected_hour = st.sidebar.slider("시간 선택 (Hour)", 0, 23, 12, format="%d:00")
time_mode = st.sidebar.radio("기준 시간", ["STD (계획)", "RAM (실제)"])
col_name = 'STD_Full' if "STD" in time_mode else 'RAM_Full'

# 데이터 필터링
filtered = data[
    (data[col_name].dt.date == selected_date) & 
    (data[col_name].dt.hour == selected_hour)
]

# ==========================================
# 3. 지도 및 통계
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

    # 빈 주기장 (회색 점)
    all_spots = pd.read_csv('rksi_stands_zoned.csv')
    occupied_spots = filtered['SPT'].unique()
    empty_spots = all_spots[~all_spots['Stand_ID'].astype(str).isin(occupied_spots)]
    
    for _, row in empty_spots.iterrows():
        folium.CircleMarker(
            [row['Lat'], row['Lon']], radius=2, color='gray', fill=True, fill_opacity=0.3,
            popup=f"Stand {row['Stand_ID']}"
        ).add_to(m)

    # 점유된 주기장 (빨간 비행기)
    for _, row in filtered.iterrows():
        popup_html = f"""
        <b>Flight:</b> {row['FLT']}<br>
        <b>Spot:</b> {row['SPT']}<br>
        <b>Dest:</b> {row['DES']}<br>
        <b>Time:</b> {row[col_name].strftime('%H:%M')}
        """
        folium.Marker(
            [row['Lat'], row['Lon']],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row['FLT']}",
            icon=folium.Icon(color='red', icon='plane', prefix='fa')
        ).add_to(m)

    st_folium(m, width="100%", height=600)

with col2:
    st.subheader(f"📊 {selected_date} {selected_hour}시")
    st.metric("출발 항공편", f"{len(filtered)} 편")
    st.caption(f"기준: {time_mode}")
    
    st.divider()
    st.write("📋 **Flight List**")
    if not filtered.empty:
        disp_cols = ['FLT', 'SPT', 'DES', 'STD' if "STD" in time_mode else 'RAM']
        st.dataframe(filtered[disp_cols].sort_values('SPT'), hide_index=True)
    else:
        st.info("해당 시간대 항공편 없음")
