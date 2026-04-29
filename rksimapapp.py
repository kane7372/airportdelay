import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import numpy as np

st.set_page_config(page_title="Incheon Airport Ultimate Dashboard", layout="wide")

# ==========================================
# 1. Load Preprocessed Data (초고속 로딩)
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists('master_dashboard_data.csv'):
        return None, "🚨 'master_dashboard_data.csv' 파일이 없습니다. 먼저 데이터 전처리 스크립트를 실행해주세요!"
    
    # df = pd.read_csv('master_dashboard_data.csv', parse_dates=['STD_Full', 'RAM_Full'])
    
    df = pd.read_parquet('master_dashboard_data.parquet')
    df['Date_Only'] = pd.to_datetime(df['Date_Only']).dt.date
    
    # [신규 추가] 지연시간 대비 지상이동시간(Taxi) 비율 계산을 위한 사전 작업
    # 지연된 항공편(Is_Delayed)이면서 지연시간이 0보다 큰 경우의 데이터만 추출
    df['Delayed_Total_Time'] = np.where(df['Is_Delayed'] & (df['Total_Delay'] > 0), df['Total_Delay'], 0)
    df['Delayed_Taxi_Time'] = np.where(df['Is_Delayed'] & (df['Total_Delay'] > 0), df['Taxi_Time'], 0)
    
    return df, "Success"

flights, msg = load_data()

if flights is None:
    st.error(msg)
    st.stop()

# ==========================================
# 2. UI Layout
# ==========================================
st.title("🛫 인천공항 통계 기반 지연 분석 (Fast Track)")

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
        Avg_Taxi_In=('Taxi_In', 'mean'),
        Sum_Delay=('Delayed_Total_Time', 'sum'),
        Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
    ).reset_index()
    
    # 비율(%) 계산: (지상이동시간 합 / 총 지연시간 합) * 100
    monthly_stats['Taxi_Ratio'] = np.where(monthly_stats['Sum_Delay'] > 0, 
                                          (monthly_stats['Sum_Taxi_Delay'] / monthly_stats['Sum_Delay']) * 100, 0)
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(monthly_stats, x='YM', y='Flight_Count', color='STS_Detail', barmode='stack', title="월별 출/도착 운항 편수"), use_container_width=True)
    with c2: st.plotly_chart(px.line(monthly_stats, x='YM', y='Delay_Count', color='STS_Detail', markers=True, title="월별 지연/결항 발생 건수"), use_container_width=True)
    
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(px.bar(monthly_stats, x='YM', y='Avg_Delay_Time', color='STS_Detail', barmode='group', title="지연 항공편 월평균 지연 시간(분)"), use_container_width=True)
    with c4:
        melt_t = monthly_stats.melt(id_vars=['YM', 'STS_Detail'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
        st.plotly_chart(px.line(melt_t, x='YM', y='Time', color='STS_Detail', line_dash='Taxi_Type', markers=True, title="월평균 지상이동시간"), use_container_width=True)

    st.divider()
    st.subheader("💡 지연 요인 분석: 전체 지연시간 중 지상이동(Taxi)이 차지하는 비중")
    st.plotly_chart(px.line(monthly_stats, x='YM', y='Taxi_Ratio', color='STS_Detail', markers=True, 
                            title="월별 지연시간 대비 지상이동시간 비중 (%)", 
                            labels={'Taxi_Ratio': '지상이동시간 비중(%)'}), use_container_width=True)

# ------------------------------------------
# [TAB 2] 일별 통계 (Daily)
# ------------------------------------------
with tab2:
    st.header("📆 일별 운항 및 지연 트렌드")
    daily_stats = flights.groupby(['Date_Only', 'STS_Detail', 'Snow_Status']).agg(
        Flight_Count=('FLT', 'count'), 
        Delay_Count=('Is_Delayed', 'sum'),
        Avg_Taxi_Out=('Taxi_Out', 'mean'), 
        Avg_Taxi_In=('Taxi_In', 'mean'),
        Sum_Delay=('Delayed_Total_Time', 'sum'),
        Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
    ).reset_index()
    
    daily_stats['Taxi_Ratio'] = np.where(daily_stats['Sum_Delay'] > 0, 
                                        (daily_stats['Sum_Taxi_Delay'] / daily_stats['Sum_Delay']) * 100, 0)
    
    st.subheader("❄️ 강설 여부에 따른 일별 평균 지상이동시간")
    melt_d = daily_stats.melt(id_vars=['Date_Only', 'STS_Detail', 'Snow_Status'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
    fig_d = px.line(melt_d, x='Date_Only', y='Time', color='STS_Detail', line_dash='Taxi_Type', facet_row='Snow_Status', markers=True, height=600)
    fig_d.update_yaxes(matches=None)
    st.plotly_chart(fig_d, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(daily_stats, x='Date_Only', y='Delay_Count', color='STS_Detail', title="일별 지연 및 결항 건수", barmode='stack'), use_container_width=True)
    with c2: st.plotly_chart(px.line(daily_stats, x='Date_Only', y='Taxi_Ratio', color='STS_Detail', markers=True, title="일별 지연시간 중 지상이동시간 비중(%)"), use_container_width=True)

# ------------------------------------------
# [TAB 3] 시간대별 통계 (Hourly)
# ------------------------------------------
with tab3:
    st.header("⏰ 시간대별(0~23시) 병목 현상 및 종합 기상 연동 분석")
    min_date, max_date = flights['Date_Only'].min(), flights['Date_Only'].max()
    sel_date_range = st.date_input("분석 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(sel_date_range) == 2:
        start_d, end_d = sel_date_range
        f_hour = flights[(flights['Date_Only'] >= start_d) & (flights['Date_Only'] <= end_d)]
        
        h_stats = f_hour.groupby(['Hour', 'STS_Detail']).agg(
            Flight_Count=('FLT', 'count'), 
            Avg_Taxi_Out=('Taxi_Out', 'mean'), 
            Avg_Taxi_In=('Taxi_In', 'mean'),
            Sum_Delay=('Delayed_Total_Time', 'sum'),
            Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
        ).reset_index()
        
        h_stats['Taxi_Ratio'] = np.where(h_stats['Sum_Delay'] > 0, 
                                        (h_stats['Sum_Taxi_Delay'] / h_stats['Sum_Delay']) * 100, 0)
        
        w_hour = f_hour.groupby('Hour').agg(
            Avg_Temp=('Temp', 'mean'), Avg_Dew=('Dew_Point', 'mean'), Avg_Vis=('Visibility', 'mean'), Avg_Wind=('Wind_Spd', 'mean'), Avg_Precip=('Precip', 'mean')
        ).reset_index()
        
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.bar(h_stats, x='Hour', y='Flight_Count', color='STS_Detail', barmode='stack', title="시간대별 스케줄 집중도"), use_container_width=True)
        with c2:
            melt_h = h_stats.melt(id_vars=['Hour', 'STS_Detail'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
            st.plotly_chart(px.line(melt_h, x='Hour', y='Time', color='STS_Detail', line_dash='Taxi_Type', markers=True, title="시간대별 평균 지상 이동시간"), use_container_width=True)
        
        st.subheader("🚥 시간대별 지연 인과율")
        st.plotly_chart(px.area(h_stats, x='Hour', y='Taxi_Ratio', color='STS_Detail', title="해당 시간대 지연시간 중 지상이동(Taxi)이 차지하는 비중 (%)"), use_container_width=True)
            
        st.divider()
        selected_weather = st.multiselect("🌤️ 기상 지표 선택 (다중 선택)", options=["기온 (°C)", "이슬점 온도 (°C)", "시정 (m)", "풍속 (KT)", "강수량 (mm)"], default=["기온 (°C)", "시정 (m)"])
        w_map = {"기온 (°C)": "Avg_Temp", "이슬점 온도 (°C)": "Avg_Dew", "시정 (m)": "Avg_Vis", "풍속 (KT)": "Avg_Wind", "강수량 (mm)": "Avg_Precip"}
        
        if selected_weather:
            fig_w = make_subplots(specs=[[{"secondary_y": True}]])
            colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
            for i, w_name in enumerate(selected_weather):
                col = w_map[w_name]
                is_sec = w_name in ["시정 (m)", "강수량 (mm)"]
                if w_name == "강수량 (mm)": fig_w.add_trace(go.Bar(x=w_hour['Hour'], y=w_hour[col], name=w_name, marker_color=colors[i]), secondary_y=is_sec)
                else: fig_w.add_trace(go.Scatter(x=w_hour['Hour'], y=w_hour[col], name=w_name, line=dict(color=colors[i])), secondary_y=is_sec)
            fig_w.update_layout(title="시간대별 평균 기상 트렌드", hovermode="x unified")
            fig_w.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            st.plotly_chart(fig_w, use_container_width=True)

# ------------------------------------------
# [TAB 4] 상세 지도 분석
# ------------------------------------------
with tab4:
    st.header("🗺️ 상세 지연 인과 및 지도 시각화")
    st.sidebar.header("지도 세부 설정")
    sel_date = st.sidebar.date_input("지도 표시 날짜", min_date, min_value=min_date, max_value=max_date)
    sel_hour = st.sidebar.slider("지도 표시 시간", 0, 23, 12)
    time_basis = st.sidebar.radio("지도 기준 시간", ["STD (계획)", "RAM (푸시백)"], index=1)
    
    t_col = "STD_Full" if time_basis == "STD (계획)" else "RAM_Full"
    map_flights = flights.dropna(subset=[t_col, 'Lat'])
    map_flights = map_flights[(map_flights[t_col].dt.date == sel_date) & (map_flights[t_col].dt.hour == sel_hour)].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("해당 시간 편수", f"{len(map_flights)}편")
    if not map_flights.empty:
        c2.metric("기온 / 이슬점", f"{map_flights['Temp'].iloc[0]}°C / {map_flights['Dew_Point'].iloc[0]}°C")
        c3.metric("풍속 / 풍향", f"{map_flights['Wind_Spd'].iloc[0]}KT / {map_flights['Wind_Dir'].iloc[0]}°")
        c4.metric("기상 (WMO)", map_flights['Weather_Desc'].iloc[0])

    m = folium.Map(location=[37.46, 126.44], zoom_start=13)
    runways = {'33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363), '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)}
    for r, c in runways.items(): folium.Marker(c, popup=r, icon=folium.Icon(color='gray', icon='plane')).add_to(m)

    c_dict = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange', 'Cancelled (CNL)': 'black'}
    for _, row in map_flights.iterrows():
        popup = f"<b>{row['FLT']} ({row['STS_Detail']})</b><br>Delay: {row['Delay_Cause']}<br>Total: {row['Total_Delay']:.0f}m"
        folium.Marker([row['Lat'], row['Lon']], popup=popup, tooltip=row['FLT'], icon=folium.Icon(color=c_dict.get(row['Delay_Cause'], 'blue'), icon='plane')).add_to(m)

    st_folium(m, width="100%", height=600)
