import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import altair as alt
import os
import glob

st.set_page_config(page_title="Incheon Airport Delay Analysis Pro", layout="wide")

# ==========================================
# 1. 데이터 로드 및 지연 상세 계산
# ==========================================
@st.cache_data
def load_data():
    file_zone = 'rksi_stands_zoned.csv'
    if not os.path.exists(file_zone): return None, "Zone file not found"
    
    ramp_files = glob.glob('*RAMP*.csv')
    if not ramp_files: return None, "No RAMP files found"
    
    df_list = []
    for f in ramp_files:
        try:
            d = pd.read_csv(f)
            df_list.append(d)
        except: pass
            
    df_flight = pd.concat(df_list, ignore_index=True)
    df_zone = pd.read_csv(file_zone)
    
    df_flight['SPT'] = df_flight['SPT'].astype(str)
    df_zone['Stand_ID'] = df_zone['Stand_ID'].astype(str)
    df_flight['Date'] = df_flight['Date'].astype(str)
    
    # 시간 파싱
    def parse_dt(date_str, time_str):
        try: return pd.to_datetime(f"20{date_str} {time_str}", format='%Y%m%d %H:%M')
        except: return pd.NaT

    df_flight['STD_Full'] = df_flight.apply(lambda x: parse_dt(x['Date'], x['STD']), axis=1)
    
    # RAM 시간 및 지연(Delay) 계산
    def calc_metrics(row):
        std = row['STD_Full']
        if pd.isna(std): return pd.NaT, 0, 0
        
        # 1. Ramp Out Time (RAM)
        try:
            ram_time = pd.to_datetime(row['RAM'], format='%H:%M').time()
            ram_dt = std.replace(hour=ram_time.hour, minute=ram_time.minute)
            # 날짜 보정
            if std.hour < 4 and ram_dt.hour > 20: ram_dt -= timedelta(days=1)
            elif std.hour > 20 and ram_dt.hour < 4: ram_dt += timedelta(days=1)
        except:
            return pd.NaT, 0, 0
            
        # 2. Ramp Delay (주기장 대기 지연): RAM - STD
        ramp_delay = (ram_dt - std).total_seconds() / 60
        
        # 3. Taxi Time (지상 이동 시간): ATD-RAM (J열)
        # J열 이름이 'ATD-RAM'이라고 가정 (또는 숫자인지 확인)
        taxi_time = 0
        if 'ATD-RAM' in row:
            try: taxi_time = float(row['ATD-RAM'])
            except: taxi_time = 0
            
        return ram_dt, ramp_delay, taxi_time

    # Apply 결과를 세 컬럼으로 분리
    res = df_flight.apply(calc_metrics, axis=1, result_type='expand')
    df_flight['RAM_Full'] = res[0]
    df_flight['Ramp_Delay'] = res[1]
    df_flight['Taxi_Time'] = res[2]
    
    # 지연 원인 분류 (Delay Classification)
    # 기준: Taxi Time이 30분 이상이면 지상이동 지연, 아니면 주기장 지연 (둘다 해당하면 더 큰 쪽)
    def classify_delay(row):
        # 지연이 거의 없는 경우 (Ramp < 15 and Taxi < 25) -> Normal
        if row['Ramp_Delay'] < 15 and row['Taxi_Time'] < 25:
            return 'Normal'
            
        # 지연 원인 판단
        # 1. Ramp Delay가 압도적으로 큰 경우
        if row['Ramp_Delay'] >= 15 and row['Taxi_Time'] < 30:
            return 'Ramp (Gate)'
        # 2. Taxi Time이 긴 경우
        elif row['Taxi_Time'] >= 30:
            # 둘 다 긴 경우 더 심각한 쪽
            if row['Ramp_Delay'] > (row['Taxi_Time'] - 20): # Taxi 기본 20분 제외하고 비교
                return 'Ramp (Gate)'
            else:
                return 'Taxi (Ground)'
        else:
            return 'Ramp (Gate)' # 기본적으로 Ramp Delay로 간주

    df_flight['Delay_Cause'] = df_flight.apply(classify_delay, axis=1)

    # 좌표 병합
    df_merged = pd.merge(df_flight, df_zone, left_on='SPT', right_on='Stand_ID', how='inner')
    
    return df_merged, "Success"

data, msg = load_data()

# ==========================================
# 2. UI 및 필터
# ==========================================
st.title("🛫 지연 원인 심층 분석 (Ramp vs Taxi)")

if data is None:
    st.error(msg)
    st.stop()

# 사이드바
st.sidebar.header("설정")
min_dt, max_dt = data['STD_Full'].min(), data['STD_Full'].max()
sel_date = st.sidebar.date_input("날짜 선택", min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
sel_hour = st.sidebar.slider("시간대 선택", 0, 23, 12)

# 하루치 데이터 (차트용)
day_data = data[data['STD_Full'].dt.date == sel_date].copy()
# 시간대 데이터 (지도용)
map_data = day_data[day_data['STD_Full'].dt.hour == sel_hour].copy()

# ==========================================
# 3. 메인: 산점도 분석 (Scatter Plot)
# ==========================================
st.subheader(f"📈 {sel_date} 지연 분포 (Ramp vs Taxi)")

col_chart, col_desc = st.columns([3, 1])

with col_chart:
    # Altair Scatter Plot
    # X축: Ramp Delay, Y축: Taxi Time
    # 색상: Delay Cause
    scatter = alt.Chart(day_data).mark_circle(size=60).encode(
        x=alt.X('Ramp_Delay', title='주기장 지연 (분)'),
        y=alt.Y('Taxi_Time', title='지상 이동 시간 (분)'),
        color=alt.Color('Delay_Cause', 
                        scale=alt.Scale(domain=['Normal', 'Ramp (Gate)', 'Taxi (Ground)'],
                                        range=['green', 'red', 'orange']),
                        legend=alt.Legend(title="지연 원인")),
        tooltip=['FLT', 'SPT', 'DES', 'Ramp_Delay', 'Taxi_Time', 'Delay_Cause']
    ).properties(height=400).interactive()
    
    # 기준선 (Taxi 30분, Ramp 15분)
    rule_taxi = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y')
    rule_ramp = alt.Chart(pd.DataFrame({'x': [15]})).mark_rule(color='gray', strokeDash=[3,3]).encode(x='x')
    
    st.altair_chart(scatter + rule_taxi + rule_ramp, use_container_width=True)

with col_desc:
    st.markdown("#### 💡 분석 가이드")
    st.info("**X축 (주기장 지연):** 출발 예정 시간보다 얼마나 늦게 램프를 떠났는지 나타냅니다.")
    st.warning("**Y축 (지상 이동):** 램프 아웃 후 이륙까지 걸린 시간입니다. 30분 이상이면 혼잡으로 봅니다.")
    st.markdown("---")
    st.write(f"**총 항공편:** {len(day_data)}편")
    st.write(f"🔴 **주기장 지연:** {len(day_data[day_data['Delay_Cause']=='Ramp (Gate)'])}편")
    st.write(f"🟠 **이동 지연:** {len(day_data[day_data['Delay_Cause']=='Taxi (Ground)'])}편")

# ==========================================
# 4. 지도 시각화
# ==========================================
st.divider()
st.subheader(f"🗺️ {sel_hour}시 주기장 현황 (원인별 색상)")

m = folium.Map(location=[37.46, 126.44], zoom_start=13)

# 활주로
runways = {'33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363), '33R': (37.4563, 126.4647), '15L': (37.4838, 126.4402)}
for r, c in runways.items():
    folium.Marker(c, popup=r, icon=folium.Icon(color='gray', icon='plane')).add_to(m)

# 마커 색상 매핑
color_map = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange'}

for _, row in map_data.iterrows():
    cause = row['Delay_Cause']
    color = color_map.get(cause, 'blue')
    
    popup_html = f'''
    <b>{row['FLT']}</b> ({cause})<br>
    Spot: {row['SPT']}<br>
    Ramp Delay: {row['Ramp_Delay']:.0f}m<br>
    Taxi Time: {row['Taxi_Time']:.0f}m
    '''
    
    folium.Marker(
        [row['Lat'], row['Lon']],
        popup=folium.Popup(popup_html, max_width=200),
        tooltip=f"{row['FLT']} ({cause})",
        icon=folium.Icon(color=color, icon='plane', prefix='fa')
    ).add_to(m)

# 범례 (Legend)
legend_html = '''
 <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 100px; 
 border:2px solid grey; z-index:9999; font-size:14px; background-color:white; padding: 10px;">
 <b>지연 원인</b><br>
 <i class="fa fa-plane" style="color:green"></i> 정상 (Normal)<br>
 <i class="fa fa-plane" style="color:red"></i> 주기장 지연 (Ramp)<br>
 <i class="fa fa-plane" style="color:orange"></i> 이동 지연 (Taxi)
 </div>
 '''
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width="100%", height=600)
