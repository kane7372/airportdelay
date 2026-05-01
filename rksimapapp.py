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
    file_path = 'master_dashboard_data.parquet'
    if not os.path.exists(file_path):
        file_path = 'master_dashboard_data.csv'
        if not os.path.exists(file_path):
            return None, "🚨 'master_dashboard_data.parquet' 파일이 없습니다. 전처리 스크립트를 먼저 실행해주세요!"
    
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, parse_dates=['STD_Full', 'RAM_Full'])
            
        if 'Date_Only' in df.columns:
            df['Date_Only'] = pd.to_datetime(df['Date_Only']).dt.date
            
        df['Delayed_Total_Time'] = np.where(df['Is_Delayed'] & (df['Total_Delay'] > 0), df['Total_Delay'], 0)
        df['Delayed_Taxi_Time'] = np.where(df['Is_Delayed'] & (df['Total_Delay'] > 0), df['Taxi_Time'], 0)
        
        domestic_fsc = ['KAL', 'AAR']
        domestic_lcc = ['JJA', 'JNA', 'TWB', 'ABL', 'ASV', 'ESR', 'APZ', 'FGW', 'ARO']
        
        def categorize_airline(code):
            if pd.isna(code) or code == 'UNK': return '분류 불명 (Unknown)'
            if code in domestic_fsc: return '국내 FSC (대형사)'
            elif code in domestic_lcc: return '국내 LCC (저비용)'
            else: return '외항사 (Foreign)'
            
        df['Airline_Group'] = df['Airline'].apply(categorize_airline)
        
        return df, "Success"
    except Exception as e:
        return None, f"🚨 데이터 로딩 중 오류가 발생했습니다: {str(e)}"

flights_raw, msg = load_data()

if flights_raw is None:
    st.error(msg)
    st.stop()

# ==========================================
# 🌟 좌측 사이드바 글로벌 필터 
# ==========================================
st.sidebar.header("🎯 통합 데이터 필터")

# 0. [신규 추가] 이상치 필터 토글 스위치
st.sidebar.markdown("### 🧹 데이터 클렌징 필터")
exclude_outliers = st.sidebar.checkbox(
    "비정상 지상이동시간 제외 (3-Sigma)", 
    value=True, 
    help="체크를 해제하면 24시간(1440분) 등 입력 오류로 추정되는 극단적인 이상치를 포함하여 원본 그대로 분석합니다."
)
st.sidebar.divider()

# 1. 여객/화물 필터
available_pax_cgo = sorted(flights_raw['Pax_Cgo'].dropna().unique().tolist())
selected_pax_cgo = st.sidebar.multiselect("1️⃣ 여객/화물 구분 (Pax/Cgo)", options=available_pax_cgo, default=available_pax_cgo)

# 2. 운항 상태 필터
available_sts = sorted(flights_raw['STS_Detail'].dropna().unique().tolist())
selected_sts = st.sidebar.multiselect("2️⃣ 운항 상태 상세 (STS)", options=available_sts, default=available_sts)

st.sidebar.divider()

# 3. 항공사 그룹 연동 필터
st.sidebar.markdown("### ✈️ 항공사 선택 (그룹 연동)")
available_groups = ["국내 FSC (대형사)", "국내 LCC (저비용)", "외항사 (Foreign)", "분류 불명 (Unknown)"]
active_groups = [g for g in available_groups if g in flights_raw['Airline_Group'].unique()]

selected_groups = st.sidebar.multiselect("3️⃣-A. 항공사 그룹 선택", options=active_groups, default=active_groups)

if selected_groups:
    filtered_airline_list = sorted(flights_raw[flights_raw['Airline_Group'].isin(selected_groups)]['Airline'].dropna().unique().tolist())
else:
    filtered_airline_list = []

selected_airlines = st.sidebar.multiselect(
    "3️⃣-B. 개별 항공사 선택", 
    options=filtered_airline_list, 
    default=filtered_airline_list
)

st.sidebar.divider()

# 4. 강설 영향권 필터
st.sidebar.markdown("### ❄️ 강설 여파(Snow Impact) 필터")
available_snow = sorted(flights_raw['Snow_Status'].dropna().unique().tolist())
selected_snow = st.sidebar.multiselect(
    "4️⃣ 강설 영향권 선택", 
    options=available_snow, 
    default=available_snow
)

# ==========================================
# 데이터 필터링 적용 (Filtered Data)
# ==========================================
if not selected_pax_cgo: selected_pax_cgo = available_pax_cgo
if not selected_sts: selected_sts = available_sts
if not selected_airlines: selected_airlines = filtered_airline_list 
if not selected_snow: selected_snow = available_snow

# 🌟 이상치 제외 여부에 따른 마스터 데이터셋 구축
flights = flights_raw[
    (flights_raw['Pax_Cgo'].isin(selected_pax_cgo)) & 
    (flights_raw['STS_Detail'].isin(selected_sts)) &
    (flights_raw['Airline'].isin(selected_airlines)) &
    (flights_raw['Snow_Status'].isin(selected_snow))
].copy()

# 토글 스위치가 켜져있고, Outlier 컬럼이 존재하면 제외 실행!
if exclude_outliers and 'Is_Taxi_Outlier' in flights.columns:
    flights = flights[flights['Is_Taxi_Outlier'] == False]

# ==========================================
# 2. UI Layout & Tabs
# ==========================================
st.title("🛫 인천공항 통계 기반 지연 분석 (Ultimate)")

if flights.empty:
    st.warning("⚠️ 선택하신 필터 조건에 맞는 데이터가 없습니다. 좌측 사이드바 필터를 변경해 주세요.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 1. 월별 통계", 
    "📆 2. 일별 통계", 
    "⏰ 3. 시간대별 통계", 
    "🗺️ 4. 상세 지도 분석",
    "✈️ 5. 항공사별 통계"
])

# ------------------------------------------
# [TAB 1] 월별 통계
# ------------------------------------------
with tab1:
    st.header("📅 월별 운항 및 지연 트렌드")
    monthly_stats = flights.groupby(['YM', 'STS_Detail']).agg(
        Flight_Count=('FLT', 'count'), Delay_Count=('Is_Delayed', 'sum'),
        Avg_Delay_Time=('Total_Delay', lambda x: x[x > 15].mean() if len(x[x > 15]) > 0 else 0),
        Avg_Taxi_Out=('Taxi_Out', 'mean'), Avg_Taxi_In=('Taxi_In', 'mean'),
        Sum_Delay=('Delayed_Total_Time', 'sum'), Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
    ).reset_index()
    monthly_stats['Taxi_Ratio'] = np.where(monthly_stats['Sum_Delay'] > 0, (monthly_stats['Sum_Taxi_Delay'] / monthly_stats['Sum_Delay']) * 100, 0)
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(monthly_stats, x='YM', y='Flight_Count', color='STS_Detail', barmode='stack', title="월별 출/도착 편수"), use_container_width=True)
    with c2: st.plotly_chart(px.line(monthly_stats, x='YM', y='Delay_Count', color='STS_Detail', markers=True, title="월별 지연/결항 건수"), use_container_width=True)
    
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(px.bar(monthly_stats, x='YM', y='Avg_Delay_Time', color='STS_Detail', barmode='group', title="지연 항공편 월평균 지연 시간(분)"), use_container_width=True)
    with c4:
        melt_t = monthly_stats.melt(id_vars=['YM', 'STS_Detail'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
        if not melt_t.empty: st.plotly_chart(px.line(melt_t, x='YM', y='Time', color='STS_Detail', line_dash='Taxi_Type', markers=True, title="월평균 지상이동시간"), use_container_width=True)

    st.subheader("💡 전체 지연시간 중 지상이동(Taxi)이 차지하는 비중")
    st.plotly_chart(px.line(monthly_stats, x='YM', y='Taxi_Ratio', color='STS_Detail', markers=True, title="월별 지연시간 대비 지상이동시간 비중 (%)"), use_container_width=True)

# ------------------------------------------
# [TAB 2] 일별 통계
# ------------------------------------------
with tab2:
    st.header("📆 일별 운항 및 지연 트렌드")
    daily_stats = flights.groupby(['Date_Only', 'STS_Detail', 'Snow_Status']).agg(
        Flight_Count=('FLT', 'count'), Delay_Count=('Is_Delayed', 'sum'),
        Avg_Taxi_Out=('Taxi_Out', 'mean'), Avg_Taxi_In=('Taxi_In', 'mean'),
        Sum_Delay=('Delayed_Total_Time', 'sum'), Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
    ).reset_index()
    daily_stats['Taxi_Ratio'] = np.where(daily_stats['Sum_Delay'] > 0, (daily_stats['Sum_Taxi_Delay'] / daily_stats['Sum_Delay']) * 100, 0)
    
    st.subheader("❄️ 기상(강설) 여부에 따른 평균 지상이동시간 요약")
    snow_summary = flights.groupby('Snow_Status').agg(
        Flight_Count=('FLT', 'count'),
        Avg_Taxi_Out=('Taxi_Out', 'mean'),
        Avg_Taxi_In=('Taxi_In', 'mean')
    ).reset_index()
    
    st.dataframe(
        snow_summary.style.format({
            'Flight_Count': '{:,.0f} 편',
            'Avg_Taxi_Out': '{:.1f} 분',
            'Avg_Taxi_In': '{:.1f} 분'
        }),
        use_container_width=True
    )
    
    melt_d = daily_stats.melt(id_vars=['Date_Only', 'STS_Detail', 'Snow_Status'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
    if not melt_d.empty:
        fig_d = px.line(melt_d, x='Date_Only', y='Time', color='STS_Detail', line_dash='Taxi_Type', facet_row='Snow_Status', markers=True, height=800, title="일별 강설 여부에 따른 평균 지상이동시간 추이")
        fig_d.update_yaxes(matches=None)
        st.plotly_chart(fig_d, use_container_width=True)
        
        with st.expander("📅 일별 상세 통계 표 보기 (클릭하여 펼치기)"):
            st.dataframe(
                daily_stats[['Date_Only', 'STS_Detail', 'Snow_Status', 'Flight_Count', 'Avg_Taxi_Out', 'Avg_Taxi_In']].sort_values('Date_Only').style.format({
                    'Avg_Taxi_Out': '{:.1f} 분',
                    'Avg_Taxi_In': '{:.1f} 분'
                }),
                use_container_width=True
            )
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(daily_stats, x='Date_Only', y='Delay_Count', color='STS_Detail', title="일별 지연 건수", barmode='stack'), use_container_width=True)
    with c2: st.plotly_chart(px.line(daily_stats, x='Date_Only', y='Taxi_Ratio', color='STS_Detail', markers=True, title="일별 지연시간 중 지상이동 비중(%)"), use_container_width=True)

# ------------------------------------------
# [TAB 3] 시간대별 통계
# ------------------------------------------
with tab3:
    st.header("⏰ 시간대별 병목 현상 및 기상 연동 분석")
    min_date, max_date = flights['Date_Only'].min(), flights['Date_Only'].max()
    # 달력 위젯 (여기서 선택한 기간이 아래 모든 차트와 '표'에 연동됨)
    sel_date_range = st.date_input("분석 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(sel_date_range) == 2:
        start_d, end_d = sel_date_range
        # 선택된 기간으로 데이터 필터링
        f_hour = flights[(flights['Date_Only'] >= start_d) & (flights['Date_Only'] <= end_d)]
        
        if not f_hour.empty:
            h_stats = f_hour.groupby(['Hour', 'STS_Detail']).agg(
                Flight_Count=('FLT', 'count'), Avg_Taxi_Out=('Taxi_Out', 'mean'), Avg_Taxi_In=('Taxi_In', 'mean'),
                Sum_Delay=('Delayed_Total_Time', 'sum'), Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
            ).reset_index()
            h_stats['Taxi_Ratio'] = np.where(h_stats['Sum_Delay'] > 0, (h_stats['Sum_Taxi_Delay'] / h_stats['Sum_Delay']) * 100, 0)
            
            w_hour = f_hour.groupby('Hour').agg(
                Avg_Temp=('Temp', 'mean'), Avg_Dew=('Dew_Point', 'mean'), Avg_Vis=('Visibility', 'mean'), Avg_Wind=('Wind_Spd', 'mean'), Avg_Precip=('Precip', 'mean')
            ).reset_index()
            
            c1, c2 = st.columns(2)
            with c1: 
                fig_h1 = px.bar(h_stats, x='Hour', y='Flight_Count', color='STS_Detail', barmode='stack', title="시간대별 스케줄 집중도")
                fig_h1.update_xaxes(tickmode='linear', dtick=1)
                st.plotly_chart(fig_h1, use_container_width=True)
            with c2:
                melt_h = h_stats.melt(id_vars=['Hour', 'STS_Detail'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
                if not melt_h.empty:
                    fig_h2 = px.line(melt_h, x='Hour', y='Time', color='STS_Detail', line_dash='Taxi_Type', markers=True, title="시간대별 평균 지상이동시간")
                    fig_h2.update_xaxes(tickmode='linear', dtick=1)
                    st.plotly_chart(fig_h2, use_container_width=True)
            
            st.plotly_chart(px.area(h_stats, x='Hour', y='Taxi_Ratio', color='STS_Detail', title="시간대 지연시간 중 지상이동 비중 (%)"), use_container_width=True)
            
            # =========================================================
            # 🌟 [신규 추가] 달력 연동: 일별 & 강설 영향권별 상세 표
            # =========================================================
            st.divider()
            st.subheader("📅 선택 기간 내 일별 & 강설 영향권별 지상이동시간 상세 표")
            st.markdown("달력에서 선택한 기간 동안의 하루하루를 **'눈 내린 상황'**에 따라 쪼개어 보여줍니다. (시간이 오래 걸릴수록 붉은색으로 표시됩니다.)")
            
            daily_snow_tab3 = f_hour.groupby(['Date_Only', 'Snow_Status']).agg(
                Flight_Count=('FLT', 'count'),
                Avg_Taxi_Out=('Taxi_Out', 'mean'),
                Avg_Taxi_In=('Taxi_In', 'mean')
            ).reset_index()
            
            # 날짜순, 그리고 강설 심각도 순으로 정렬
            daily_snow_tab3 = daily_snow_tab3.sort_values(by=['Date_Only', 'Snow_Status'])
            
            # 표 렌더링 (컬러 그라데이션 적용)
            st.dataframe(
                daily_snow_tab3.style.format({
                    'Flight_Count': '{:,.0f} 편',
                    'Avg_Taxi_Out': '{:.1f} 분',
                    'Avg_Taxi_In': '{:.1f} 분'
                }).background_gradient(
                    subset=['Avg_Taxi_Out', 'Avg_Taxi_In'], 
                    cmap='OrRd' # 주황-빨강 히트맵 색상
                ),
                use_container_width=True
            )
            st.divider()
            # =========================================================

            selected_weather = st.multiselect("🌤️ 기상 지표 선택", options=["기온 (°C)", "이슬점 온도 (°C)", "시정 (m)", "풍속 (KT)", "강수량 (mm)"], default=["기온 (°C)", "시정 (m)"])
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
                fig_w.update_xaxes(tickmode='linear', dtick=1)
                st.plotly_chart(fig_w, use_container_width=True)
# ------------------------------------------
# [TAB 4] 상세 지도 분석
# ------------------------------------------
with tab4:
    st.header("🗺️ 상세 지연 인과 및 지도 시각화")
    st.sidebar.header("지도 세부 설정 (Tab 4 전용)")
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

    # 인천공항 중심 좌표
    m = folium.Map(location=[37.46, 126.44], zoom_start=14)
    
    # 1. 활주로(Runway) 마커 추가 (회색 비행기)
    runways = {'33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363), '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)}
    for r, c in runways.items(): 
        folium.Marker(c, tooltip=f"Runway {r}", icon=folium.Icon(color='lightgray', icon='road', prefix='fa')).add_to(m)

    # 🌟 2. [신규 추가] 제방빙장(De-icing Pad) 마커 고정 표시 (눈꽃 아이콘)
    try:
        # 주기장 데이터셋을 다시 로드하여 제방빙장 위치만 추출
        zone_file = 'rksi_stands_zoned.csv'
        if os.path.exists('rksi_stands_zoned (2).csv'): zone_file = 'rksi_stands_zoned (2).csv'
        df_pads = pd.read_csv(zone_file)
        
        # 'De-icing Apron' 카테고리만 필터링
        deice_pads = df_pads[df_pads['Category'].astype(str).str.contains('De-icing', case=False, na=False)]
        
        for _, pad in deice_pads.iterrows():
            folium.CircleMarker(
                location=[pad['Lat'], pad['Lon']],
                radius=5,                 # 🌟 원의 크기 (숫자를 줄이면 더 작아집니다)
                color='#1E90FF',          # 테두리 색상 (진한 파란색)
                weight=2,                 # 테두리 두께
                fill=True,                # 원 안을 색으로 채움
                fill_color='#87CEFA',     # 채우기 색상 (연한 파란색)
                fill_opacity=0.7,         # 투명도
                tooltip=f"❄️ 제방빙장 (Stand {pad['Stand_ID']})"
            ).add_to(m)
    except Exception as e:
        pass # 파일이 없거나 오류가 나면 무시하고 패스

    # 3. 항공편(Flight) 마커 추가
    c_dict = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange', 'Cancelled (CNL)': 'black'}
    for _, row in map_flights.iterrows():
        is_dep = row['Flight_Dir'] == 'DEP'
        dir_label = "🛫 출발" if is_dep else "🛬 도착"
        fa_icon = "plane-departure" if is_dep else "plane-arrival"
        
        popup = f"<b>{row['FLT']} ({dir_label})</b><br>{row['Airline_Group']} | {row['STS_Detail']}<br>Delay: {row['Delay_Cause']}<br>Total: {row['Total_Delay']:.0f}m"
        tooltip_text = f"{row['FLT']} ({dir_label})"
        
        folium.Marker(
            [row['Lat'], row['Lon']], 
            popup=popup, 
            tooltip=tooltip_text, 
            icon=folium.Icon(color=c_dict.get(row['Delay_Cause'], 'blue'), icon=fa_icon, prefix='fa')
        ).add_to(m)

    st_folium(m, width="100%", height=650)
# ------------------------------------------
# [TAB 5] 항공사별 통계
# ------------------------------------------
with tab5:
    st.header("✈️ 항공사별 종합 운영 퍼포먼스")
    st.markdown("항공사 그룹 및 개별 항공사별 운항 점유율, 지연율, 지상이동 효율성을 비교합니다.")
    
    airline_stats = flights.groupby(['Airline_Group', 'Airline']).agg(
        Flight_Count=('FLT', 'count'),
        Delay_Count=('Is_Delayed', 'sum'),
        Avg_Delay_Time=('Total_Delay', lambda x: x[x > 15].mean() if len(x[x > 15]) > 0 else 0),
        Avg_Taxi_Out=('Taxi_Out', 'mean'),
        Avg_Taxi_In=('Taxi_In', 'mean')
    ).reset_index()
    
    airline_stats['Delay_Rate(%)'] = (airline_stats['Delay_Count'] / airline_stats['Flight_Count']) * 100
    airline_stats = airline_stats.sort_values('Flight_Count', ascending=False)
    
    top_n = st.slider("그래프에 표시할 상위 항공사 수 (운항 편수 기준)", min_value=5, max_value=len(airline_stats) if len(airline_stats) > 0 else 5, value=min(20, len(airline_stats)))
    view_stats = airline_stats.head(top_n)
    
    if not view_stats.empty:
        fig_a1 = px.bar(view_stats, x='Airline', y=['Flight_Count', 'Delay_Count'], 
                        title=f"상위 {top_n}개 항공사 운항 편수 대비 지연 건수", barmode='group',
                        labels={'value': '건수 (편)', 'variable': '구분'})
        st.plotly_chart(fig_a1, use_container_width=True)
        
        fig_a2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_a2.add_trace(go.Bar(x=view_stats['Airline'], y=view_stats['Avg_Delay_Time'], name="평균 지연시간 (분)", marker_color='#FFA15A'), secondary_y=False)
        fig_a2.add_trace(go.Scatter(x=view_stats['Airline'], y=view_stats['Delay_Rate(%)'], name="지연율 (%)", mode='lines+markers', line=dict(color='#EF553B', width=3)), secondary_y=True)
        fig_a2.update_layout(title=f"상위 {top_n}개 항공사 평균 지연시간 및 지연율", hovermode="x unified")
        fig_a2.update_yaxes(title_text="평균 지연시간 (분)", secondary_y=False)
        fig_a2.update_yaxes(title_text="지연율 (%)", secondary_y=True)
        st.plotly_chart(fig_a2, use_container_width=True)
        
        melted_taxi = view_stats.melt(id_vars=['Airline', 'Airline_Group'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
        if not melted_taxi.empty:
            fig_a3 = px.bar(melted_taxi, x='Airline', y='Time', color='Taxi_Type', barmode='group', 
                            title=f"상위 {top_n}개 항공사 평균 지상이동시간", hover_data=['Airline_Group'])
            st.plotly_chart(fig_a3, use_container_width=True)
    
    st.divider()
    st.subheader("📊 항공사별 상세 통계 표")
    st.dataframe(
        airline_stats.style.format({
            'Avg_Delay_Time': '{:.1f} 분',
            'Avg_Taxi_Out': '{:.1f} 분',
            'Avg_Taxi_In': '{:.1f} 분',
            'Delay_Rate(%)': '{:.1f} %'
        }), 
        use_container_width=True
    )
