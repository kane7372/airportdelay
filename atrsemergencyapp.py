import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import numpy as np

st.set_page_config(page_title="Incheon Airport Departure Dashboard", layout="wide")

# ==========================================
# 1. Load Preprocessed Data
# ==========================================
@st.cache_data
def load_data():
    file_path = 'master_dashboard_data2.parquet'
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
            df['Date_Only'] = pd.to_datetime(df['Date_Only'], errors='coerce').dt.date
            df['Date_Only'] = df['Date_Only'].fillna('날짜미상')            
            
        df['Delayed_Total_Time'] = np.where(df['Is_Delayed'] & (df['Total_Delay'] > 0), df['Total_Delay'], 0)
        df['Delayed_Taxi_Time'] = np.where(df['Is_Delayed'] & (df['Total_Delay'] > 0), df['Taxi_Time'], 0)
        
        # 항공사 그룹 분류
        domestic_fsc = ['KAL', 'AAR']
        domestic_lcc = ['JJA', 'JNA', 'TWB', 'ABL', 'ASV', 'ESR', 'APZ', 'FGW', 'ARO']
        
        def categorize_airline(code):
            if pd.isna(code) or code == 'UNK': return '분류 불명 (Unknown)'
            if code in domestic_fsc: return '국내 FSC (대형사)'
            elif code in domestic_lcc: return '국내 LCC (저비용)'
            else: return '외항사 (Foreign)'
            
        df['Airline_Group'] = df['Airline'].apply(categorize_airline)
        
        # 🌟 [수정 반영] 마스터 데이터의 고도화된 Weather_Desc를 매핑
        def categorize_weather(w_desc):
            w = str(w_desc)
            if '건설' in w: return '건설 (Dry Snow)'
            elif '습설' in w: return '습설 (Wet Snow)'
            elif '날림눈' in w or '눈보라' in w: return '건설 (Dry Snow)'
            # 📌 안개/빙무 조건이 포함되도록 조건 추가 보완
            elif '어는 비' in w or '결빙' in w or '진눈깨비' in w or '안개/빙무' in w: return '기타 제방빙 위험기상'
            else: return '비강설 (Non-Snow)'
            
        df['Snow_Type'] = df['Weather_Desc'].apply(categorize_weather)
        
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

st.sidebar.markdown("### 🧹 데이터 클렌징 필터")
exclude_outliers = st.sidebar.checkbox(
    "비정상 지상이동시간 제외 (3-Sigma)", 
    value=True, 
    help="체크를 해제하면 24시간(1440분) 등 입력 오류로 추정되는 극단적인 이상치를 포함하여 원본 그대로 분석합니다."
)
st.sidebar.divider()

available_pax_cgo = sorted(flights_raw['Pax_Cgo'].dropna().unique().tolist())
selected_pax_cgo = st.sidebar.multiselect("1️⃣ 여객/화물 구분 (Pax/Cgo)", options=available_pax_cgo, default=available_pax_cgo)

available_sts = sorted(flights_raw['STS_Detail'].dropna().unique().tolist())
selected_sts = st.sidebar.multiselect("2️⃣ 출발 운항 상태 (STS2 기준)", options=available_sts, default=available_sts)

st.sidebar.divider()

st.sidebar.markdown("### ✈️ 항공사 선택 (그룹 연동)")
available_groups = ["국내 FSC (대형사)", "국내 LCC (저비용)", "외항사 (Foreign)", "분류 불명 (Unknown)"]
active_groups = [g for g in available_groups if g in flights_raw['Airline_Group'].unique()]

selected_groups = st.sidebar.multiselect("3️⃣-A. 항공사 그룹 선택", options=active_groups, default=active_groups)

if selected_groups:
    filtered_airline_list = sorted(flights_raw[flights_raw['Airline_Group'].isin(selected_groups)]['Airline'].dropna().unique().tolist())
else:
    filtered_airline_list = []

selected_airlines = st.sidebar.multiselect("3️⃣-B. 개별 항공사 선택", options=filtered_airline_list, default=filtered_airline_list)

st.sidebar.divider()

st.sidebar.markdown("### ❄️ 강설 여파(Snow Impact) 필터")
available_snow = sorted(flights_raw['Snow_Status'].dropna().unique().tolist())
selected_snow = st.sidebar.multiselect("4️⃣ 강설 영향권 선택", options=available_snow, default=available_snow)

# ==========================================
# 데이터 필터링 적용
# ==========================================
if not selected_pax_cgo: selected_pax_cgo = available_pax_cgo
if not selected_sts: selected_sts = available_sts
if not selected_airlines: selected_airlines = filtered_airline_list 
if not selected_snow: selected_snow = available_snow

flights = flights_raw[
    (flights_raw['Pax_Cgo'].isin(selected_pax_cgo)) & 
    (flights_raw['STS_Detail'].isin(selected_sts)) &
    (flights_raw['Airline'].isin(selected_airlines)) &
    (flights_raw['Snow_Status'].isin(selected_snow))
].copy()

if exclude_outliers and 'Is_Taxi_Outlier' in flights.columns:
    flights = flights[flights['Is_Taxi_Outlier'] == False]

if 'YM' in flights.columns:
    flights['YM'] = flights['YM'].fillna('날짜미상')

is_cnl = flights['STS_Detail'].str.contains('결항|cnl', case=False, na=False)
flights.loc[is_cnl, 'Total_Delay'] = flights.loc[is_cnl, 'Total_Delay'].fillna(0)
flights.loc[is_cnl, 'Taxi_Out'] = flights.loc[is_cnl, 'Taxi_Out'].fillna(0)

# ==========================================
# 2. UI Layout & Tabs
# ==========================================
st.title("🛫 인천공항 출발(DEP) 통계 및 지연 분석 대시보드")

if flights.empty:
    st.warning("⚠️ 선택하신 필터 조건에 맞는 데이터가 없습니다. 좌측 사이드바 필터를 변경해 주세요.")
    st.stop()

total_count = len(flights_raw)
filtered_count = len(flights)
st.info(f"📊 **현재 설정된 필터 기준 데이터:** 총 **{filtered_count:,}** 건 (전체 {total_count:,} 건)")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📅 1. 월별 통계", 
    "📆 2. 일별 통계", 
    "⏰ 3. 시간대별 통계", 
    "🗺️ 4. 상세 지도 분석",
    "✈️ 5. 항공사별 통계",
    "❄️ 6. 제방빙 및 강설 심층 분석"
])

# ------------------------------------------
# [TAB 1] 월별 통계
# ------------------------------------------
with tab1:
    st.header("📅 월별 출발 운항 및 지연 트렌드")
    monthly_stats = flights.groupby(['YM', 'STS_Detail'], dropna=False).agg(
        Flight_Count=('FLT', 'count'), Delay_Count=('Is_Delayed', 'sum'),
        Avg_Delay_Time=('Total_Delay', lambda x: x[x > 15].mean() if len(x[x > 15]) > 0 else 0),
        Avg_Total_Delay=('Total_Delay', 'mean'), 
        Avg_Taxi_Out=('Taxi_Out', 'mean'),
        Sum_Delay=('Delayed_Total_Time', 'sum'), Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
    ).reset_index()    
    
    monthly_stats['Delay_to_Taxi_Ratio'] = np.where(
        monthly_stats['Sum_Taxi_Delay'] > 0, 
        (monthly_stats['Sum_Delay'] / monthly_stats['Sum_Taxi_Delay']) * 100, 
        np.nan 
    )
    
    c1, c2 = st.columns(2)
    with c1: 
        fig_c1 = px.bar(monthly_stats, x='YM', y='Flight_Count', color='STS_Detail', barmode='stack', title="월별 출발 편수")
        fig_c1.update_xaxes(type='category', categoryorder='category ascending')
        st.plotly_chart(fig_c1, use_container_width=True)
        
    with c2: 
        bad_stats = monthly_stats[
            (monthly_stats['Delay_Count'] > 0) | 
            (monthly_stats['STS_Detail'].str.contains('결항|cnl', case=False, na=False))
        ]
        fig_c2 = px.line(bad_stats, x='YM', y='Flight_Count', color='STS_Detail', markers=True, title="월별 지연/결항 건수")
        fig_c2.update_xaxes(type='category', categoryorder='category ascending')
        st.plotly_chart(fig_c2, use_container_width=True)        
    
    st.divider()
    with st.expander("📊 월별 출발 및 운항 상태 상세 데이터 표 보기"):
        tc1, tc2 = st.columns(2)
        
        with tc1:
            st.markdown("**✈️ 월별 출발 편수**")
            pivot_dep = flights.groupby(['YM', 'STS_Detail']).size().unstack(fill_value=0)
            if not pivot_dep.empty:
                pivot_dep['총합'] = pivot_dep.sum(axis=1)
                pivot_dep.index = pivot_dep.index.astype(str)
                st.dataframe(pivot_dep.style.background_gradient(cmap='Blues'), use_container_width=True)            
                
        with tc2:
            st.markdown("**🚨 월별 운항 상태 (15분 이상 지연/결항 등) 건수**")
            bad_flights = flights[
                (flights['Is_Delayed'] == True) | 
                (flights['STS_Detail'].str.contains('결항|cnl', case=False, na=False))
            ]
            pivot_sts = bad_flights.groupby(['YM', 'STS_Detail']).size().unstack(fill_value=0)
            if not pivot_sts.empty:
                pivot_sts['총합'] = pivot_sts.sum(axis=1)
                pivot_sts.index = pivot_sts.index.astype(str) + "월"
                st.dataframe(pivot_sts.style.background_gradient(cmap='OrRd'), use_container_width=True)   

    c3, c4 = st.columns(2)
    with c3: 
        c3_stats = monthly_stats[~monthly_stats['STS_Detail'].str.contains('CNL|DLA', na=False)]
        fig_c3 = px.bar(c3_stats, x='YM', y='Avg_Total_Delay', color='STS_Detail', barmode='group', title="정상 항공편 단순 평균 지연 시간(분)")
        fig_c3.update_xaxes(type='category', categoryorder='category ascending')
        st.plotly_chart(fig_c3, use_container_width=True)
        
    with c4:
        fig_c4 = px.bar(monthly_stats, x='YM', y='Avg_Delay_Time', color='STS_Detail', barmode='group', title="지연 항공편 월평균 지연 시간(분)")
        fig_c4.update_xaxes(type='category', categoryorder='category ascending')
        st.plotly_chart(fig_c4, use_container_width=True)

    fig_taxi_out = px.bar(monthly_stats, x='YM', y='Avg_Taxi_Out', color='STS_Detail', barmode='group', title="월별 평균 Taxi-Out 소요시간(분)")
    fig_taxi_out.update_xaxes(type='category', categoryorder='category ascending')
    st.plotly_chart(fig_taxi_out, use_container_width=True)

    st.subheader("💡 지상이동(Taxi) 지연시간 대비 전체 지연시간 비율")
    fig_bar = px.bar(
        monthly_stats, 
        x='YM', 
        y='Delay_to_Taxi_Ratio', 
        color='STS_Detail', 
        barmode='group', 
        title="월별 지상이동 지연시간 대비 전체 지연시간 비율 (%)"
    )
    fig_bar.update_xaxes(type='category', categoryorder='category ascending')
    st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------
# [TAB 2] 일별 통계
# ------------------------------------------
with tab2:
    st.header("📆 일별 출발 운항 및 지연 트렌드")
    daily_stats = flights.groupby(['Date_Only', 'STS_Detail', 'Snow_Status']).agg(
        Flight_Count=('FLT', 'count'), Delay_Count=('Is_Delayed', 'sum'),
        Avg_Taxi_Out=('Taxi_Out', 'mean'),
        Sum_Delay=('Delayed_Total_Time', 'sum'), Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
    ).reset_index()
    daily_stats['Taxi_Ratio'] = np.where(daily_stats['Sum_Delay'] > 0, (daily_stats['Sum_Taxi_Delay'] / daily_stats['Sum_Delay']) * 100, 0)
    
    st.subheader("❄️ 기상(강설) 여부에 따른 출발 지상이동시간(Taxi-Out) 운영 임계점 분석")
    snow_summary = flights.groupby('Snow_Status').agg(
        Flight_Count=('FLT', 'count'), Avg_Taxi_Out=('Taxi_Out', 'mean'), Std_Taxi_Out=('Taxi_Out', 'std')
    ).reset_index().fillna(0)
    
    snow_summary['Taxi_Out_2Sig'] = snow_summary['Avg_Taxi_Out'] + (2 * snow_summary['Std_Taxi_Out'])
    snow_summary['Taxi_Out_3Sig'] = snow_summary['Avg_Taxi_Out'] + (3 * snow_summary['Std_Taxi_Out'])
    
    display_summary = snow_summary[['Snow_Status', 'Flight_Count', 'Avg_Taxi_Out', 'Taxi_Out_2Sig', 'Taxi_Out_3Sig']]
    
    st.dataframe(
        display_summary.style.format({
            'Flight_Count': '{:,.0f} 편', 'Avg_Taxi_Out': '{:.1f} 분 (평균)',
            'Taxi_Out_2Sig': '{:.1f} 분 (경고선)', 'Taxi_Out_3Sig': '🚨 {:.1f} 분 (마비선)'
        }).background_gradient(subset=['Taxi_Out_3Sig'], cmap='Reds'),
        use_container_width=True
    )    

    if not daily_stats.empty:
        fig_d = px.line(daily_stats, x='Date_Only', y='Avg_Taxi_Out', color='STS_Detail', facet_row='Snow_Status', markers=True, height=800, title="일별 강설 여부에 따른 평균 Taxi-Out 추이")
        # 🌟 일별 데이터 가독성 및 날짜 공백 연도 축소 최적화 적용
        fig_d.update_xaxes(type='category', categoryorder='category ascending')
        fig_d.update_yaxes(matches=None)
        st.plotly_chart(fig_d, use_container_width=True)
        
        with st.expander("📅 일별 상세 통계 표 보기"):
            st.dataframe(
                daily_stats[['Date_Only', 'STS_Detail', 'Snow_Status', 'Flight_Count', 'Avg_Taxi_Out']].sort_values('Date_Only').style.format({
                    'Avg_Taxi_Out': '{:.1f} 분'
                }),
                use_container_width=True
            )
    
    c1, c2 = st.columns(2)
    with c1: 
        fig_daily_delay = px.bar(daily_stats, x='Date_Only', y='Delay_Count', color='STS_Detail', title="일별 지연 건수", barmode='stack')
        fig_daily_delay.update_xaxes(type='category', categoryorder='category ascending')
        st.plotly_chart(fig_daily_delay, use_container_width=True)
    
    daily_stats['Taxi_Ratio'] = daily_stats['Taxi_Ratio'].clip(upper=100)
    with c2: 
        c2_stats = daily_stats[daily_stats['Taxi_Ratio'] > 0]
        fig_daily_ratio = px.line(c2_stats, x='Date_Only', y='Taxi_Ratio', color='STS_Detail', markers=True, title="일별 지연시간 중 지상이동 비중(%)")
        fig_daily_ratio.update_xaxes(type='category', categoryorder='category ascending')
        st.plotly_chart(fig_daily_ratio, use_container_width=True)

# ------------------------------------------
# [TAB 3] 시간대별 통계
# ------------------------------------------
with tab3:
    st.header("⏰ 시간대별 병목 현상 및 기상 연동 분석")
    min_date, max_date = flights['Date_Only'].min(), flights['Date_Only'].max()
    sel_date_range = st.date_input("분석 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(sel_date_range) == 2:
        start_d, end_d = sel_date_range
        f_hour = flights[(flights['Date_Only'] >= start_d) & (flights['Date_Only'] <= end_d)]
        
        if not f_hour.empty:
            h_stats = f_hour.groupby(['Hour', 'STS_Detail']).agg(
                Flight_Count=('FLT', 'count'), Avg_Taxi_Out=('Taxi_Out', 'mean'),
                Sum_Delay=('Delayed_Total_Time', 'sum'), Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
            ).reset_index()
            h_stats['Taxi_Ratio'] = np.where(h_stats['Sum_Delay'] > 0, (h_stats['Sum_Taxi_Delay'] / h_stats['Sum_Delay']) * 100, 0)
            
            c1, c2 = st.columns(2)
            with c1: 
                fig_h1 = px.bar(h_stats, x='Hour', y='Flight_Count', color='STS_Detail', barmode='stack', title="시간대별 스케줄 집중도")
                fig_h1.update_xaxes(tickmode='linear', dtick=1)
                st.plotly_chart(fig_h1, use_container_width=True)
            with c2:
                fig_h2 = px.line(h_stats, x='Hour', y='Avg_Taxi_Out', color='STS_Detail', markers=True, title="시간대별 평균 Taxi-Out 소요시간")
                fig_h2.update_xaxes(tickmode='linear', dtick=1)
                st.plotly_chart(fig_h2, use_container_width=True)
            
            def get_weather_summary(x):
                w_list = [str(w) for w in x.dropna().unique() if str(w) not in ['-', 'UNK']]
                if not w_list: return '알 수 없음'
                severe_w = [w for w in w_list if w != '일반']
                return ', '.join(severe_w) if severe_w else '일반 (맑음)'

            with st.expander("⏰ 시간대별 스케줄 집중도 및 운항 상태 표 보기"):
                pivot_h = h_stats.pivot(index='Hour', columns='STS_Detail', values='Flight_Count').fillna(0).astype(int)
                pivot_h['총 운항편수'] = pivot_h.sum(axis=1)
                hourly_weather = f_hour.groupby('Hour').agg(Avg_Temp=('Temp', 'mean'), Weather_Info=('Weather_Desc', get_weather_summary))
                pivot_h = pivot_h.join(hourly_weather).rename(columns={'Avg_Temp': '평균 기온', 'Weather_Info': '주요 기상 현상'})
                pivot_h.index = pivot_h.index.astype(str).str.zfill(2) + "시"
                st.dataframe(pivot_h.style.format({'평균 기온': '{:.1f} °C'}).background_gradient(subset=['총 운항편수'], cmap='Blues'), use_container_width=True)
            
            with st.expander("📅 선택 기간 내 일별 & 강설 영향권별 지상이동시간 상세 표 보기"):
                daily_snow_tab3 = f_hour.groupby(['Date_Only', 'Snow_Status']).agg(
                    Flight_Count=('FLT', 'count'), Avg_Taxi_Out=('Taxi_Out', 'mean'), Avg_Temp=('Temp', 'mean'), Weather_Info=('Weather_Desc', get_weather_summary)
                ).reset_index().sort_values(by=['Date_Only', 'Snow_Status'])
                daily_snow_tab3 = daily_snow_tab3.rename(columns={'Weather_Info': '주요 기상 현상', 'Avg_Temp': '평균 기온'})
                st.dataframe(
                    daily_snow_tab3.style.format({
                        'Flight_Count': '{:,.0f} 편', 'Avg_Taxi_Out': '{:.1f} 분', '평균 기온': '{:.1f} °C'
                    }).background_gradient(subset=['Avg_Taxi_Out'], cmap='OrRd'),
                    use_container_width=True
                )
            st.divider()

# ------------------------------------------
# [TAB 4] 상세 지도 및 기상 분석 통합
# ------------------------------------------
with tab4:
    st.header("🗺️ 상세 지연 인과 및 지도/기상 시각화")
    
    st.sidebar.header("지도 세부 설정 (Tab 4 전용)")
    sel_date = st.sidebar.date_input("조회 날짜", min_date, min_value=min_date, max_value=max_date)
    sel_hour = st.sidebar.slider("지도 표시 시간", 0, 23, 12)
    time_basis = st.sidebar.radio("지도 기준 시간", ["STD (계획)", "RAM (푸시백)"], index=1)
    
    t_col = "STD_Full" if time_basis == "STD (계획)" else "RAM_Full"
    
    f_day = flights[pd.to_datetime(flights['Date_Only']).dt.date == sel_date].copy()
    map_flights = f_day[f_day[t_col].dt.hour == sel_hour].copy()
    map_flights = map_flights.dropna(subset=['Lat', 'Lon'])

    with st.expander("🗺️ 선택 시간대 공항 상세 지도 보기", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("해당 시간 편수", f"{len(map_flights)}편")
        if not map_flights.empty:
            c2.metric("기온 / 이슬점", f"{map_flights['Temp'].iloc[0]}°C / {map_flights['Dew_Point'].iloc[0]}°C")
            c3.metric("풍속 / 습도", f"{map_flights['Wind_Spd'].iloc[0]}KT / {map_flights.get('Humidity', pd.Series([np.nan])).iloc[0]}%")
            c4.metric("기상 (WMO)", map_flights['Weather_Desc'].iloc[0])

        m = folium.Map(location=[37.46, 126.44], zoom_start=14)
        runways = {'33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363), '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)}
        for r, c in runways.items(): folium.Marker(c, tooltip=f"Runway {r}", icon=folium.Icon(color='lightgray', icon='road', prefix='fa')).add_to(m)

        try:
            zone_file = 'rksi_stands_zoned (2).csv' if os.path.exists('rksi_stands_zoned (2).csv') else 'rksi_stands_zoned.csv'
            df_pads = pd.read_csv(zone_file)
            deice_pads = df_pads[df_pads['Category'].astype(str).str.contains('De-icing', case=False, na=False)].dropna(subset=['Lat', 'Lon'])
            for _, pad in deice_pads.iterrows():
                folium.CircleMarker(
                    location=[pad['Lat'], pad['Lon']], radius=5, color='#1E90FF', fill=True,
                    fill_color='#87CEFA', fill_opacity=0.7, tooltip=f"❄️ 제방빙장 (Stand {pad['Stand_ID']})"
                ).add_to(m)
        except: pass

        c_dict = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange', 'Cancelled (CNL)': 'black'}
        for _, row in map_flights.iterrows():
            fa_icon = "plane-departure"
            popup = f"<b>{row['FLT']}</b><br>Delay: {row['Delay_Cause']}<br>Total: {row['Total_Delay']:.0f}m"
            folium.Marker(
                [row['Lat'], row['Lon']], popup=popup, tooltip=row['FLT'],
                icon=folium.Icon(color=c_dict.get(row['Delay_Cause'], 'blue'), icon=fa_icon, prefix='fa')
            ).add_to(m)
        st_folium(m, width="100%", height=600)

    st.divider()
    st.subheader(f"☀️ {sel_date} 기상 및 운영 상세 분석")

    f_hour = f_day.copy()
    w_hour = f_hour.groupby('Hour').agg(
        Avg_Temp=('Temp', 'mean'), Avg_Dew=('Dew_Point', 'mean'), 
        Avg_Vis=('Visibility', 'mean'), Avg_Wind=('Wind_Spd', 'mean'), 
        Avg_Precip=('Precip', 'mean'), Avg_Hum=('Humidity', 'mean')
    ).reset_index()

    with st.expander("⏰ 시간대별 운항 현황 & 기상 통합 분석", expanded=True):
        pivot_sts = f_hour.groupby(['Hour', 'Snow_Status', 'STS_Detail']).size().unstack(fill_value=0)
        metrics_base = f_hour.groupby(['Hour', 'Snow_Status']).agg(
            총_편수=('FLT', 'count'), Avg_Taxi_Out=('Taxi_Out', 'mean'), Avg_Temp=('Temp', 'mean'), 
            Avg_Hum=('Humidity', 'mean'), Weather_Info=('Weather_Desc', get_weather_summary)
        )
        combined_df = metrics_base.join(pivot_sts).reset_index().sort_values(by=['Hour', 'Snow_Status'])
        combined_df['Hour'] = combined_df['Hour'].astype(int).astype(str).str.zfill(2) + "시"
        combined_df = combined_df.rename(columns={'Hour': '시간대', 'Snow_Status': '강설 영향권', 'Weather_Info': '주요 기상 현상', 'Avg_Temp': '평균 기온', 'Avg_Hum': '평균 상대습도', 'Avg_Taxi_Out': 'Taxi-Out(평균)'})

        f_col1, f_col2 = st.columns(2)
        with f_col1: sel_hours = st.multiselect("⏰ 시간대 필터", options=combined_df['시간대'].unique())
        with f_col2: sel_snow = st.multiselect("❄️ 강설 영향권 필터", options=combined_df['강설 영향권'].unique())

        filtered_df = combined_df.copy()
        if sel_hours: filtered_df = filtered_df[filtered_df['시간대'].isin(sel_hours)]
        if sel_snow: filtered_df = filtered_df[filtered_df['강설 영향권'].isin(sel_snow)]

        if not filtered_df.empty:
            status_cols = [c for c in pivot_sts.columns if c in filtered_df.columns]
            st.dataframe(
                filtered_df.style.format({'총 편수': '{:,.0f} 편', 'Taxi-Out(평균)': '{:.1f} 분', '평균 기온': '{:.1f} °C', '평균 상대습도': '{:.1f} %'})
                .background_gradient(subset=['총_편수'] + status_cols, cmap='Blues')
                .background_gradient(subset=['Taxi-Out(평균)'], cmap='OrRd'),
                use_container_width=True
            )
        else: st.warning("조건에 맞는 데이터가 없습니다.")

    with st.expander("📉 강설 영향권별 Taxi-Out 회복 곡선 (Recovery Curve)"):
        recovery_stats = f_hour.groupby('Snow_Status').agg(Avg_Taxi_Out=('Taxi_Out', 'mean')).reset_index()
        if not recovery_stats.empty:
            fig_rec = px.line(recovery_stats, x='Snow_Status', y='Avg_Taxi_Out', markers=True, text='Avg_Taxi_Out', title="회복 단계별 평균 Taxi-Out 추이")
            fig_rec.update_traces(texttemplate='%{text:.1f}분', textposition='top center')
            st.plotly_chart(fig_rec, use_container_width=True)

    selected_weather = st.multiselect("🌤️ 시각화할 기상 지표 선택", options=["기온 (°C)", "이슬점 온도 (°C)", "상대습도 (%)", "시정 (m)", "풍속 (KT)", "강수량 (mm)"], default=["기온 (°C)", "상대습도 (%)"])
    w_map = {"기온 (°C)": "Avg_Temp", "이슬점 온도 (°C)": "Avg_Dew", "상대습도 (%)": "Avg_Hum", "시정 (m)": "Avg_Vis", "풍속 (KT)": "Avg_Wind", "강수량 (mm)": "Avg_Precip"}
    
    if selected_weather:
        fig_w = make_subplots(specs=[[{"secondary_y": True}]])
        colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
        for i, w_name in enumerate(selected_weather):
            col = w_map[w_name]
            is_sec = w_name in ["시정 (m)", "강수량 (mm)", "상대습도 (%)"]
            if w_name == "강수량 (mm)": fig_w.add_trace(go.Bar(x=w_hour['Hour'], y=w_hour[col], name=w_name, marker_color=colors[i]), secondary_y=is_sec)
            else: fig_w.add_trace(go.Scatter(x=w_hour['Hour'], y=w_hour[col], name=w_name, line=dict(color=colors[i])), secondary_y=is_sec)
        fig_w.update_layout(title=f"{sel_date} 시간대별 기상 트렌드", hovermode="x unified")
        st.plotly_chart(fig_w, use_container_width=True)

# ------------------------------------------
# [TAB 5] 항공사별 통계
# ------------------------------------------
with tab5:
    st.header("✈️ 항공사별 종합 운영 퍼포먼스")
    airline_stats = flights.groupby(['Airline_Group', 'Airline']).agg(
        Flight_Count=('FLT', 'count'), Delay_Count=('Is_Delayed', 'sum'),
        Avg_Delay_Time=('Total_Delay', lambda x: x[x > 15].mean() if len(x[x > 15]) > 0 else 0),
        Avg_Taxi_Out=('Taxi_Out', 'mean')
    ).reset_index()
    
    airline_stats['Delay_Rate(%)'] = (airline_stats['Delay_Count'] / airline_stats['Flight_Count']) * 100
    airline_stats = airline_stats.sort_values('Flight_Count', ascending=False)
    
    top_n = st.slider("그래프에 표시할 상위 항공사 수", min_value=5, max_value=len(airline_stats) if len(airline_stats) > 0 else 5, value=min(20, len(airline_stats)))
    view_stats = airline_stats.head(top_n)
    
    if not view_stats.empty:
        fig_a1 = px.bar(view_stats, x='Airline', y=['Flight_Count', 'Delay_Count'], title=f"상위 {top_n}개 항공사 운항 편수 대비 지연 건수", barmode='group')
        st.plotly_chart(fig_a1, use_container_width=True)
        
        fig_a2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_a2.add_trace(go.Bar(x=view_stats['Airline'], y=view_stats['Avg_Delay_Time'], name="평균 지연시간 (분)", marker_color='#FFA15A'), secondary_y=False)
        fig_a2.add_trace(go.Scatter(x=view_stats['Airline'], y=view_stats['Delay_Rate(%)'], name="지연율 (%)", mode='lines+markers', line=dict(color='#EF553B', width=3)), secondary_y=True)
        fig_a2.update_layout(title=f"상위 {top_n}개 항공사 평균 지연시간 및 지연율", hovermode="x unified")
        st.plotly_chart(fig_a2, use_container_width=True)
        
        fig_a3 = px.bar(view_stats, x='Airline', y='Avg_Taxi_Out', title=f"상위 {top_n}개 항공사 평균 Taxi-Out 소요시간", hover_data=['Airline_Group'])
        st.plotly_chart(fig_a3, use_container_width=True)
        
        st.divider()
        with st.expander("📉 항공사 그룹별 강설 회복 곡선 (FSC vs LCC) 보기"):
            airline_recovery = flights.groupby(['Snow_Status', 'Airline_Group'])['Taxi_Out'].mean().reset_index()
            if not airline_recovery.empty:
                fig_a_rec = px.line(airline_recovery, x='Snow_Status', y='Taxi_Out', color='Airline_Group', markers=True, title="항공사 그룹별 강설 영향권에 따른 평균 Taxi-Out 회복 추이")
                fig_a_rec.update_traces(line=dict(width=4), marker=dict(size=10))
                st.plotly_chart(fig_a_rec, use_container_width=True)

    st.divider()
    st.subheader("📊 항공사별 상세 통계 표")
    st.dataframe(airline_stats.style.format({'Avg_Delay_Time': '{:.1f} 분', 'Avg_Taxi_Out': '{:.1f} 분', 'Delay_Rate(%)': '{:.1f} %'}), use_container_width=True)

# ------------------------------------------
# [TAB 6] 제방빙 및 강설 심층 분석
# ------------------------------------------
with tab6:
    st.header("❄️ 강설 유형 및 위험 기상에 따른 지상이동시간 심층 분석")
    st.markdown("마스터 데이터 전처리 단계에서 물리 엔진(Clausius-Clapeyron)을 통해 분류된 **건설(Dry Snow) vs 습설(Wet Snow)** 및 기타 제방빙이 필요한 위험 기상 유무에 따른 차이를 확인합니다.")
    
    snow_type_stats = flights.groupby('Snow_Type').agg(
        Flight_Count=('FLT', 'count'),
        Avg_Total_Delay=('Total_Delay', 'mean'),
        Avg_Taxi_Out=('Taxi_Out', 'mean'),
        Std_Taxi_Out=('Taxi_Out', 'std'),
        Avg_Temp=('Temp', 'mean'),
        Avg_Hum=('Humidity', 'mean')
    ).reset_index()
    
    sort_order = {"비강설 (Non-Snow)": 0, "건설 (Dry Snow)": 1, "습설 (Wet Snow)": 2, "기타 제방빙 위험기상": 3}
    snow_type_stats['Order'] = snow_type_stats['Snow_Type'].map(sort_order)
    snow_type_stats = snow_type_stats.sort_values('Order').drop(columns=['Order'])
    
    st.subheader("📊 강설 유형별 평균 지연 및 지상이동시간 비교")
    c1, c2 = st.columns(2)
    colors = {"비강설 (Non-Snow)": '#00CC96', "건설 (Dry Snow)": '#AB63FA', "습설 (Wet Snow)": '#EF553B', "기타 제방빙 위험기상": '#FFA15A'}
    
    with c1:
        fig_s1 = px.bar(snow_type_stats, x='Snow_Type', y='Avg_Taxi_Out', color='Snow_Type', title="유형별 평균 Taxi-Out 소요시간 (분)", text_auto='.1f', color_discrete_map=colors)
        st.plotly_chart(fig_s1, use_container_width=True)
        
    with c2:
        fig_s2 = px.bar(snow_type_stats, x='Snow_Type', y='Avg_Total_Delay', color='Snow_Type', title="유형별 평균 전체 지연시간 (분)", text_auto='.1f', color_discrete_map=colors)
        st.plotly_chart(fig_s2, use_container_width=True)
        
    st.divider()
    st.subheader("📈 Taxi-Out 시간 분포 및 변동성 (Box Plot)")
    st.markdown("제방빙 작업 난이도에 따른 **데이터의 산포도(Variance)와 이상치 꼬리(Tail)**를 파악합니다. 습설이나 어는 비 조건일수록 시간이 불규칙해집니다.")
    
    fig_box = px.box(flights, x='Snow_Type', y='Taxi_Out', color='Snow_Type', title="강설/기상 유형별 Taxi-Out 시간 분포", color_discrete_map=colors, hover_data=['FLT', 'Temp', 'Humidity', 'Total_Delay'])
    fig_box.update_traces(boxpoints='outliers', jitter=0.3)
    fig_box.update_xaxes(categoryorder='array', categoryarray=["비강설 (Non-Snow)", "건설 (Dry Snow)", "습설 (Wet Snow)", "기타 제방빙 위험기상"])
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.divider()
    st.subheader("📋 기상 유형별 상세 물리 지표 요약")
    st.dataframe(
        snow_type_stats.style.format({
            'Flight_Count': '{:,.0f} 편', 'Avg_Total_Delay': '{:.1f} 분',
            'Avg_Taxi_Out': '{:.1f} 분', 'Std_Taxi_Out': '{:.1f} 분',
            'Avg_Temp': '{:.1f} °C', 'Avg_Hum': '{:.1f} %'
        }).background_gradient(subset=['Avg_Taxi_Out', 'Std_Taxi_Out', 'Avg_Hum'], cmap='OrRd'),
        use_container_width=True
    )
