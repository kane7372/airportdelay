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
            # 1. errors='coerce'를 넣어서 '날짜미상' 같은 이상한 글자는 에러를 뿜지 않고 일단 빈칸(NaT)으로 만듭니다.
            df['Date_Only'] = pd.to_datetime(df['Date_Only'], errors='coerce').dt.date
            # 2. 빈칸이 된 곳에 다시 '날짜미상'이라는 꼬리표를 예쁘게 달아줍니다.
            df['Date_Only'] = df['Date_Only'].fillna('날짜미상')            
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

if 'YM' in flights.columns:
    flights['YM'] = flights['YM'].fillna('날짜미상')

is_cnl = flights['STS_Detail'].str.contains('결항|cnl', case=False, na=False)
flights.loc[is_cnl, 'Total_Delay'] = flights.loc[is_cnl, 'Total_Delay'].fillna(0)
flights.loc[is_cnl, 'Taxi_Out'] = flights.loc[is_cnl, 'Taxi_Out'].fillna(0)
flights.loc[is_cnl, 'Taxi_In'] = flights.loc[is_cnl, 'Taxi_In'].fillna(0)

# ==========================================
# 2. UI Layout & Tabs
# ==========================================
st.title("🛫 인천공항 통계 기반 지연 분석 (Ultimate)")

if flights.empty:
    st.warning("⚠️ 선택하신 필터 조건에 맞는 데이터가 없습니다. 좌측 사이드바 필터를 변경해 주세요.")
    st.stop()

# 🌟 [신규 추가] 전체 데이터 건수 및 필터링된 데이터 건수 표시
total_count = len(flights_raw) # 처음에 불러온 원본 데이터프레임 이름
filtered_count = len(flights) # 필터가 다 적용된 최종 데이터프레임 이름

st.info(f"📊 **현재 설정된 필터 기준 데이터:** 총 **{filtered_count:,}** 건 (전체 {total_count:,} 건)")

st.divider() # 시각적 분리를 위해 선 하나 추가

# ---------------------------------------------------------
# (기존 코드) 탭 생성 부분 시작
# tab1, tab2, tab3, tab4, tab5 = st.tabs([...])
# ---------------------------------------------------------

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
# 🌟 groupby에 dropna=False 를 추가하여 YM이 비어있더라도 결항 데이터가 살아남게 합니다!
    monthly_stats = flights.groupby(['YM', 'STS_Detail'], dropna=False).agg(
        Flight_Count=('FLT', 'count'), Delay_Count=('Is_Delayed', 'sum'),
        Avg_Delay_Time=('Total_Delay', lambda x: x[x > 15].mean() if len(x[x > 15]) > 0 else 0), # 기존 (15분 이상 지연 평균)
        Avg_Total_Delay=('Total_Delay', 'mean'), # 🌟 신규 추가 (전체 비행기 단순 평균)
        Avg_Taxi_Out=('Taxi_Out', 'mean'), Avg_Taxi_In=('Taxi_In', 'mean'),
        Sum_Delay=('Delayed_Total_Time', 'sum'), Sum_Taxi_Delay=('Delayed_Taxi_Time', 'sum')
    ).reset_index()    
    # 지상이동 비율 계산 및 100% 캡 씌우기
    monthly_stats['Delay_to_Taxi_Ratio'] = np.where(
    monthly_stats['Sum_Taxi_Delay'] > 0, 
    (monthly_stats['Sum_Delay'] / monthly_stats['Sum_Taxi_Delay']) * 100, 
    np.nan 
)
    monthly_stats['Delay_to_Taxi_Ratio'] = np.clip(monthly_stats['Delay_to_Taxi_Ratio'], 0, 100)
    
    c1, c2 = st.columns(2)
    with c1: 
        st.plotly_chart(px.bar(monthly_stats, x='YM', y='Flight_Count', color='STS_Detail', barmode='stack', title="월별 출/도착 편수"), use_container_width=True)
    with c2: 
        bad_stats = monthly_stats[
            (monthly_stats['Delay_Count'] > 0) | 
            (monthly_stats['STS_Detail'].str.contains('결항|cnl', case=False, na=False))
        ]
        
        # ⚠️ 중요: y축을 Delay_Count가 아닌 'Flight_Count'로 둬야 결항된 비행기 숫자도 올바르게 카운트되어 그래프가 솟아오릅니다!
        fig_c2 = px.line(bad_stats, x='YM', y='Flight_Count', color='STS_Detail', markers=True, title="월별 지연/결항 건수")
        st.plotly_chart(fig_c2, use_container_width=True)        
    
    # =========================================================
    # 🌟 [신규 추가] 월별 트렌드 상세 데이터 표 (Expander)
    # =========================================================
    st.divider()
    with st.expander("📊 월별 출/도착 및 운항 상태 상세 데이터 표 보기 (클릭하여 펼치기)"):
        st.markdown("위 그래프의 기반이 되는 월별 상세 수치입니다. 숫자가 클수록 진한 색으로 표시됩니다.")
            
        # 표를 좌우로 예쁘게 배치하기 위해 컬럼 분할 (Expander 안쪽에 위치해야 함!)
        tc1, tc2 = st.columns(2)
        
        with tc1:
            st.markdown("**✈️ 월별 출/도착 편수**")
            # 🌟 수정됨: 'Flight_Count' 대신 'ARR_DEP' (출발/도착 컬럼) 사용!
            pivot_arr_dep = flights.groupby(['YM', 'STS_Detail']).size().unstack(fill_value=0)
                
            if not pivot_arr_dep.empty:
                pivot_arr_dep['총합'] = pivot_arr_dep.sum(axis=1)
                # 인덱스(YM)를 문자열로 변환
                pivot_arr_dep.index = pivot_arr_dep.index.astype(str)
                    
                # 파란색 히트맵 적용하여 출력
                st.dataframe(
                    pivot_arr_dep.style.background_gradient(cmap='Blues'),
                    use_container_width=True
                )            
                
        with tc2:
            st.markdown("**🚨 월별 운항 상태 (15분 이상 지연/결항 등) 건수**")
            
            # 🌟 핵심 수정: 지연(Is_Delayed)이거나, 상태에 '결항' 또는 'cnl'이 포함된 경우 모두 필터링!
            bad_flights = flights[
                (flights['Is_Delayed'] == True) | 
                (flights['STS_Detail'].str.contains('결항|cnl', case=False, na=False))
            ]
            
            # 필터링된 데이터로 피벗 테이블 생성
            pivot_sts = bad_flights.groupby(['YM', 'STS_Detail']).size().unstack(fill_value=0)
            
            if not pivot_sts.empty:
                pivot_sts['총합'] = pivot_sts.sum(axis=1)
                pivot_sts.index = pivot_sts.index.astype(str) + "월"
                
                # 붉은색 히트맵 적용하여 출력
                st.dataframe(
                    pivot_sts.style.background_gradient(cmap='OrRd'),
                    use_container_width=True
                )    
    c3, c4, c5 = st.columns(3)
    with c3: 
        c3_stats = monthly_stats[~monthly_stats['STS_Detail'].str.contains('CNL|DLA', na=False)]
        st.plotly_chart(px.bar(c3_stats, x='YM', y='Avg_Total_Delay', color='STS_Detail', barmode='group', title="전체 항공편 단순 평균 지연 시간(분)"), use_container_width=True)
    with c4 : st.plotly_chart(px.bar(monthly_stats, x='YM', y='Avg_Delay_Time', color='STS_Detail', barmode='group', title="15분 이상 지연 항공편 월평균 지연 시간(분)"), use_container_width=True)
    #추세그래프
    # with c5 :
    #     melt_t = monthly_stats.melt(id_vars=['YM', 'STS_Detail'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
    #     if not melt_t.empty: st.plotly_chart(px.line(melt_t, x='YM', y='Time', color='STS_Detail', line_dash='Taxi_Type', markers=True, title="월평균 지상이동시간"), use_container_width=True)
    with c5:
        melt_t = monthly_stats.melt(id_vars=['YM', 'STS_Detail'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
    if not melt_t.empty: 
        fig = px.bar(
            melt_t, 
            x='YM', 
            y='Time', 
            color='STS_Detail', 
            facet_col='Taxi_Type',     # Taxi_Type을 기준으로 그래프를 좌우로 나눔
            barmode='group', 
            title="월평균 지상이동시간"
        )
        st.plotly_chart(fig, use_container_width=True)

st.subheader("💡 지상이동(Taxi) 지연시간 대비 전체 지연시간 비율")
fig_bar = px.bar(
    monthly_stats, 
    x='YM', 
    y='Delay_to_Taxi_Ratio', 
    color='STS_Detail', 
    barmode='group',  # 막대들을 겹치지 않고 나란히 세우기
    title="월별 지상이동 지연시간 대비 전체 지연시간 비율 (%)"
)
st.plotly_chart(fig_bar, use_container_width=True)
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
    
    st.subheader("❄️ 기상(강설) 여부에 따른 지상이동시간 운영 임계점(Threshold) 분석")
    st.markdown("각 강설 상태별 평균 시간과, 통계적 통제 한계선인 **2-Sigma(경고)** 및 **3-Sigma(심각/마비)** 상한선을 제시합니다.")
    
    # 1. 평균과 표준편차(std)를 함께 집계
    snow_summary = flights.groupby('Snow_Status').agg(
        Flight_Count=('FLT', 'count'),
        Avg_Taxi_Out=('Taxi_Out', 'mean'),
        Std_Taxi_Out=('Taxi_Out', 'std'),
        Avg_Taxi_In=('Taxi_In', 'mean'),
        Std_Taxi_In=('Taxi_In', 'std')
    ).reset_index()
    
    # NaN 값(데이터가 1개뿐이어서 표준편차가 없는 경우 등) 방어
    snow_summary = snow_summary.fillna(0)
    
    # 2. 2-Sigma, 3-Sigma 상한선(Upper Bound) 계산
    snow_summary['Taxi_Out_2Sig'] = snow_summary['Avg_Taxi_Out'] + (2 * snow_summary['Std_Taxi_Out'])
    snow_summary['Taxi_Out_3Sig'] = snow_summary['Avg_Taxi_Out'] + (3 * snow_summary['Std_Taxi_Out'])
    
    snow_summary['Taxi_In_2Sig'] = snow_summary['Avg_Taxi_In'] + (2 * snow_summary['Std_Taxi_In'])
    snow_summary['Taxi_In_3Sig'] = snow_summary['Avg_Taxi_In'] + (3 * snow_summary['Std_Taxi_In'])
    
    # 3. 화면에 보여줄 컬럼만 선택 및 정렬
    display_summary = snow_summary[[
        'Snow_Status', 'Flight_Count', 
        'Avg_Taxi_Out', 'Taxi_Out_2Sig', 'Taxi_Out_3Sig',
        'Avg_Taxi_In', 'Taxi_In_2Sig', 'Taxi_In_3Sig'
    ]]
    
    # 4. 표 렌더링 (보기 좋게 포맷팅)
    st.dataframe(
        display_summary.style.format({
            'Flight_Count': '{:,.0f} 편',
            'Avg_Taxi_Out': '{:.1f} 분 (평균)',
            'Taxi_Out_2Sig': '{:.1f} 분 (경고선)',
            'Taxi_Out_3Sig': '🚨 {:.1f} 분 (마비선)',
            'Avg_Taxi_In': '{:.1f} 분 (평균)',
            'Taxi_In_2Sig': '{:.1f} 분 (경고선)',
            'Taxi_In_3Sig': '🚨 {:.1f} 분 (마비선)'
        }).background_gradient(
            subset=['Taxi_Out_3Sig', 'Taxi_In_3Sig'], 
            cmap='Reds' # 3시그마 컬럼에 붉은색 하이라이트
        ),
        use_container_width=True
    )    
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
    
    daily_stats['Taxi_Ratio'] = daily_stats['Taxi_Ratio'].clip(upper=100)
    with c2: 
        # 1. Taxi_Ratio가 0보다 큰(0이 아닌) 진짜 지연 데이터만 뜰채로 걸러냅니다.
        c2_stats = daily_stats[daily_stats['Taxi_Ratio'] > 0]
        
        # 2. 걸러낸 c2_stats를 사용해서 차트를 그립니다.
        st.plotly_chart(px.line(c2_stats, x='Date_Only', y='Taxi_Ratio', color='STS_Detail', markers=True, title="일별 지연시간 중 지상이동 비중(%)"), use_container_width=True)

# ------------------------------------------
# [TAB 3] 시간대별 통계
# ------------------------------------------
with tab3:
    st.header("⏰ 시간대별 병목 현상 및 기상 연동 분석")
    min_date, max_date = flights['Date_Only'].min(), flights['Date_Only'].max()
    # 달력 위젯
    sel_date_range = st.date_input("분석 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(sel_date_range) == 2:
        start_d, end_d = sel_date_range
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
            
            # 🌟 [공통] 기상 현상(Weather_Desc)을 요약하기 위한 커스텀 함수
            def get_weather_summary(x):
                w_list = [str(w) for w in x.dropna().unique() if str(w) not in ['-', 'UNK']]
                if not w_list: return '알 수 없음'
                severe_w = [w for w in w_list if w != '일반']
                return ', '.join(severe_w) if severe_w else '일반 (맑음)'

            # =========================================================
            # 1. 시간대별 스케줄 집중도 표 (피벗 테이블) -> 기상 현상 추가
            # =========================================================
            with st.expander("⏰ 시간대별 스케줄 집중도 및 운항 상태 표 보기 (클릭하여 펼치기)"):
                st.markdown("선택하신 기간 동안 하루 24시간 중 **어느 시간대(Hour)에 항공편이 가장 많이 몰리는지(Peak Hour)**와 해당 시간의 기상을 나타냅니다.")
                
                # 운항 상태 피벗 테이블 생성
                pivot_h = h_stats.pivot(index='Hour', columns='STS_Detail', values='Flight_Count').fillna(0).astype(int)
                pivot_h['총 운항편수'] = pivot_h.sum(axis=1)
                
                # 시간대별 평균 기온 및 주요 기상 현상 그룹핑
                hourly_weather = f_hour.groupby('Hour').agg(
                    Avg_Temp=('Temp', 'mean'),
                    Weather_Info=('Weather_Desc', get_weather_summary)
                )
                
                # 피벗 테이블에 기상 정보 병합 및 이름 변경
                pivot_h = pivot_h.join(hourly_weather)
                pivot_h = pivot_h.rename(columns={'Avg_Temp': '평균 기온', 'Weather_Info': '주요 기상 현상'})
                
                # 인덱스 포맷팅 (00시, 01시...)
                pivot_h.index = pivot_h.index.astype(str).str.zfill(2) + "시"
                
                # 표 렌더링 (기온 포맷 및 총 운항편수 파란색 히트맵 적용)
                st.dataframe(
                    pivot_h.style.format({'평균 기온': '{:.1f} °C'}).background_gradient(subset=['총 운항편수'], cmap='Blues'),
                    use_container_width=True
                )
            
            # =========================================================
            # 2. 일별 & 강설 영향권별 상세 표 -> Expander 적용
            # =========================================================
            with st.expander("📅 선택 기간 내 일별 & 강설 영향권별 지상이동시간 상세 표 보기 (클릭하여 펼치기)"):
                st.markdown("달력에서 선택한 기간 동안의 하루하루를 **'눈 내린 상황'**에 따라 쪼개어 보여줍니다. (시간이 오래 걸릴수록 붉은색으로 표시됩니다.)")
                
                daily_snow_tab3 = f_hour.groupby(['Date_Only', 'Snow_Status']).agg(
                    Flight_Count=('FLT', 'count'),
                    Avg_Taxi_Out=('Taxi_Out', 'mean'),
                    Avg_Taxi_In=('Taxi_In', 'mean'),
                    Avg_Temp=('Temp', 'mean'),
                    Weather_Info=('Weather_Desc', get_weather_summary)
                ).reset_index()
                
                daily_snow_tab3 = daily_snow_tab3.sort_values(by=['Date_Only', 'Snow_Status'])
                daily_snow_tab3 = daily_snow_tab3.rename(columns={'Weather_Info': '주요 기상 현상', 'Avg_Temp': '평균 기온'})
                
                st.dataframe(
                    daily_snow_tab3.style.format({
                        'Flight_Count': '{:,.0f} 편',
                        'Avg_Taxi_Out': '{:.1f} 분',
                        'Avg_Taxi_In': '{:.1f} 분',
                        '평균 기온': '{:.1f} °C'
                    }).background_gradient(
                        subset=['Avg_Taxi_Out', 'Avg_Taxi_In'], 
                        cmap='OrRd' 
                    ),
                    use_container_width=True
                )
            st.divider()

            # # =========================================================
            # # 🌟 [신규 추가] 3. 강설 여파 회복 곡선 (Recovery Curve) -> Expander 적용
            # # =========================================================
            # with st.expander("📉 강설 영향권별 지상운영 회복 곡선 보기 (클릭하여 펼치기)"):
            #     st.markdown("눈이 내리는 시점(1단계)부터 공항 운영이 정상화(6단계)되기까지, **시간 경과에 따른 지상이동시간의 전체 회복 탄력성(Resilience)**을 시각화합니다.")

            #     recovery_stats = f_hour.groupby('Snow_Status').agg(
            #         Avg_Taxi_Out=('Taxi_Out', 'mean'),
            #         Avg_Taxi_In=('Taxi_In', 'mean')
            #     ).reset_index()

            #     melt_recovery = recovery_stats.melt(
            #         id_vars=['Snow_Status'], 
            #         value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], 
            #         var_name='Taxi_Type', 
            #         value_name='Time'
            #     ).dropna()

            #     if not melt_recovery.empty:
            #         fig_rec = px.line(
            #             melt_recovery, 
            #             x='Snow_Status', 
            #             y='Time', 
            #             color='Taxi_Type',
            #             markers=True, 
            #             text='Time',
            #             title="강설 영향권 6단계에 따른 평균 지상이동시간 변화 및 회복 추이",
            #             color_discrete_map={'Avg_Taxi_Out': '#EF553B', 'Avg_Taxi_In': '#00CC96'}
            #         )
                    
            #         fig_rec.update_traces(
            #             texttemplate='%{text:.1f}분', 
            #             textposition='top center', 
            #             line=dict(width=4), 
            #             marker=dict(size=12)
            #         )
            #         fig_rec.update_layout(hovermode="x unified")
            #         fig_rec.update_yaxes(title_text="평균 소요 시간 (분)")
            #         fig_rec.update_xaxes(title_text="강설 영향권 (회복 단계)")
                    
            #         st.plotly_chart(fig_rec, use_container_width=True)
            # st.divider()
            
            # selected_weather = st.multiselect("🌤️ 기상 지표 선택", options=["기온 (°C)", "이슬점 온도 (°C)", "시정 (m)", "풍속 (KT)", "강수량 (mm)"], default=["기온 (°C)", "시정 (m)"])
            # w_map = {"기온 (°C)": "Avg_Temp", "이슬점 온도 (°C)": "Avg_Dew", "시정 (m)": "Avg_Vis", "풍속 (KT)": "Avg_Wind", "강수량 (mm)": "Avg_Precip"}
            # if selected_weather:
            #     fig_w = make_subplots(specs=[[{"secondary_y": True}]])
            #     colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
            #     for i, w_name in enumerate(selected_weather):
            #         col = w_map[w_name]
            #         is_sec = w_name in ["시정 (m)", "강수량 (mm)"]
            #         if w_name == "강수량 (mm)": fig_w.add_trace(go.Bar(x=w_hour['Hour'], y=w_hour[col], name=w_name, marker_color=colors[i]), secondary_y=is_sec)
            #         else: fig_w.add_trace(go.Scatter(x=w_hour['Hour'], y=w_hour[col], name=w_name, line=dict(color=colors[i])), secondary_y=is_sec)
            #     fig_w.update_layout(title="시간대별 평균 기상 트렌드", hovermode="x unified")
            #     fig_w.update_xaxes(tickmode='linear', dtick=1)
            #     st.plotly_chart(fig_w, use_container_width=True)
# ------------------------------------------
# [TAB 4] 상세 지도 및 기상 분석 통합
# ------------------------------------------
with tab4:
    st.header("🗺️ 상세 지연 인과 및 지도/기상 시각화")
    
    # 1. 사이드바 설정 (기존 로직 유지)
    st.sidebar.header("지도 세부 설정 (Tab 4 전용)")
    sel_date = st.sidebar.date_input("조회 날짜", min_date, min_value=min_date, max_value=max_date)
    sel_hour = st.sidebar.slider("지도 표시 시간", 0, 23, 12)
    time_basis = st.sidebar.radio("지도 기준 시간", ["STD (계획)", "RAM (푸시백)"], index=1)
    
    t_col = "STD_Full" if time_basis == "STD (계획)" else "RAM_Full"
    
    # [데이터 준비] 선택한 '날짜' 전체 데이터 (기상/통계용)
    f_day = flights[pd.to_datetime(flights['Date_Only']).dt.date == sel_date].copy()
    
    map_flights = f_day[f_day[t_col].dt.hour == sel_hour].copy()
    map_flights = map_flights.dropna(subset=['Lat', 'Lon'])

    # =================================================================
    # 🌟 [수정] 지도를 접었다 펼 수 있도록 Expander 적용!
    # expanded=True 로 설정해두면 처음 탭에 들어왔을 때는 지도가 펼쳐져 있습니다.
    # =================================================================
    with st.expander("🗺️ 선택 시간대 공항 상세 지도 보기 (클릭하여 접기/펼치기)", expanded=True):
        
        # 상단 지표 (Metrics)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("해당 시간 편수", f"{len(map_flights)}편")
        if not map_flights.empty:
            c2.metric("기온 / 이슬점", f"{map_flights['Temp'].iloc[0]}°C / {map_flights['Dew_Point'].iloc[0]}°C")
            c3.metric("풍속 / 풍향", f"{map_flights['Wind_Spd'].iloc[0]}KT / {map_flights['Wind_Dir'].iloc[0]}°")
            c4.metric("기상 (WMO)", map_flights['Weather_Desc'].iloc[0])

        # --- 지도 시각화 영역 ---
        m = folium.Map(location=[37.46, 126.44], zoom_start=14)
        
        # 활주로 마커
        runways = {'33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363), '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)}
        for r, c in runways.items(): 
            folium.Marker(c, tooltip=f"Runway {r}", icon=folium.Icon(color='lightgray', icon='road', prefix='fa')).add_to(m)

        # 제방빙장 마커
        try:
            zone_file = 'rksi_stands_zoned (2).csv' if os.path.exists('rksi_stands_zoned (2).csv') else 'rksi_stands_zoned.csv'
            df_pads = pd.read_csv(zone_file)
            deice_pads = df_pads[df_pads['Category'].astype(str).str.contains('De-icing', case=False, na=False)]
            deice_pads = deice_pads.dropna(subset=['Lat', 'Lon'])
            for _, pad in deice_pads.iterrows():
                folium.CircleMarker(
                    location=[pad['Lat'], pad['Lon']], radius=5, color='#1E90FF', fill=True,
                    fill_color='#87CEFA', fill_opacity=0.7, tooltip=f"❄️ 제방빙장 (Stand {pad['Stand_ID']})"
                ).add_to(m)
        except: pass

        # 항공편 마커
        c_dict = {'Normal': 'green', 'Ramp (Gate)': 'red', 'Taxi (Ground)': 'orange', 'Cancelled (CNL)': 'black'}
        for _, row in map_flights.iterrows():
            is_dep = row['Flight_Dir'] == 'DEP'
            fa_icon = "plane-departure" if is_dep else "plane-arrival"
            popup = f"<b>{row['FLT']}</b><br>Delay: {row['Delay_Cause']}<br>Total: {row['Total_Delay']:.0f}m"
            folium.Marker(
                [row['Lat'], row['Lon']], popup=popup, tooltip=row['FLT'],
                icon=folium.Icon(color=c_dict.get(row['Delay_Cause'], 'blue'), icon=fa_icon, prefix='fa')
            ).add_to(m)

        st_folium(m, width="100%", height=600)
    # =================================================================
    # 🌟 [추가] 기상 분석 로직 (선택한 날짜 sel_date 연동)
    # =================================================================
    st.divider()
    st.subheader(f"☀️ {sel_date} 기상 및 운영 상세 분석")

    # [데이터 가공] 통계용 기초 데이터 생성
    f_hour = f_day.copy()
    h_stats = f_hour.groupby(['Hour', 'STS_Detail']).agg(Flight_Count=('FLT', 'count')).reset_index()
    w_hour = f_hour.groupby('Hour').agg(
        Avg_Temp=('Temp', 'mean'), Avg_Dew=('Dew_Point', 'mean'), 
        Avg_Vis=('Visibility', 'mean'), Avg_Wind=('Wind_Spd', 'mean'), 
        Avg_Precip=('Precip', 'mean')
    ).reset_index()

    def get_weather_summary(x):
        w_list = [str(w) for w in x.dropna().unique() if str(w) not in ['-', 'UNK']]
        if not w_list: return '알 수 없음'
        severe_w = [w for w in w_list if w != '일반']
        return ', '.join(severe_w) if severe_w else '일반 (맑음)'

    # =========================================================
    # 🌟 [통합] 시간대별 운항 상태 & 지상운영 & 기상 통합 분석 표
    # =========================================================
    with st.expander("⏰ 시간대별 운항 현황 & 기상 & 지상이동 통합 분석 (클릭하여 펼치기)", expanded=True):
        st.markdown("선택한 날짜의 시간대별로 **운항 상태(정상/지연/결항), 지상이동시간, 기상 현상**을 한눈에 분석할 수 있는 통합 지표입니다.")

        # 1. 데이터 가공 및 병합 로직
        # A. 상태별(STS_Detail) 항공편 대수 피벗 (DEP_NML, ARR_NML 등)
        pivot_sts = f_hour.groupby(['Hour', 'Snow_Status', 'STS_Detail']).size().unstack(fill_value=0)
        
        # B. 지상운영 및 기상 지표 집계
        metrics_base = f_hour.groupby(['Hour', 'Snow_Status']).agg(
            총_편수=('FLT', 'count'),
            Avg_Taxi_Out=('Taxi_Out', 'mean'),
            Avg_Taxi_In=('Taxi_In', 'mean'),
            Avg_Temp=('Temp', 'mean'),
            Weather_Info=('Weather_Desc', get_weather_summary)
        )
        
        # C. 두 데이터 결합
        combined_df = metrics_base.join(pivot_sts).reset_index()
        
        # 정렬 및 시간대 포맷팅
        combined_df = combined_df.sort_values(by=['Hour', 'Snow_Status'])
        combined_df['Hour'] = combined_df['Hour'].astype(int).astype(str).str.zfill(2) + "시"
        
        # 컬럼명 깔끔하게 정리
        combined_df = combined_df.rename(columns={
            'Hour': '시간대',
            'Snow_Status': '강설 영향권',
            'Weather_Info': '주요 기상 현상',
            'Avg_Temp': '평균 기온',
            'Avg_Taxi_Out': 'Taxi-Out(평균)',
            'Avg_Taxi_In': 'Taxi-In(평균)'
        })

        # 2. 필터 UI
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            sel_hours = st.multiselect("⏰ 시간대 필터", options=combined_df['시간대'].unique())
        with f_col2:
            sel_snow = st.multiselect("❄️ 강설 영향권 필터", options=combined_df['강설 영향권'].unique())

        # 필터 적용
        filtered_df = combined_df.copy()
        if sel_hours: filtered_df = filtered_df[filtered_df['시간대'].isin(sel_hours)]
        if sel_snow: filtered_df = filtered_df[filtered_df['강설 영향권'].isin(sel_snow)]

        # 3. 통합 표 출력 (히트맵 스타일 적용)
        if not filtered_df.empty:
            # 동적으로 생성된 상태 컬럼들 추출 (스타일링 대상)
            status_cols = [c for c in pivot_sts.columns if c in filtered_df.columns]
            
            st.dataframe(
                filtered_df.style.format({
                    '총 편수': '{:,.0f} 편',
                    'Taxi-Out(평균)': '{:.1f} 분',
                    'Taxi-In(평균)': '{:.1f} 분',
                    '평균 기온': '{:.1f} °C'
                }).background_gradient(subset=['총_편수'] + status_cols, cmap='Blues') # 운항 편수는 파란색
                  .background_gradient(subset=['Taxi-Out(평균)', 'Taxi-In(평균)'], cmap='OrRd'), # 이동 시간은 주황색
                use_container_width=True
            )
        else:
            st.warning("조건에 맞는 데이터가 없습니다.")
    # --- 그래프 1. 강설 여파 회복 곡선 ---
    with st.expander("📉 강설 영향권별 지상운영 회복 곡선 (Recovery Curve)"):
        recovery_stats = f_hour.groupby('Snow_Status').agg(Avg_Taxi_Out=('Taxi_Out', 'mean'), Avg_Taxi_In=('Taxi_In', 'mean')).reset_index()
        melt_recovery = recovery_stats.melt(id_vars=['Snow_Status'], value_vars=['Avg_Taxi_Out', 'Avg_Taxi_In'], var_name='Taxi_Type', value_name='Time').dropna()
        if not melt_recovery.empty:
            fig_rec = px.line(melt_recovery, x='Snow_Status', y='Time', color='Taxi_Type', markers=True, text='Time', title="회복 단계별 평균 지상이동시간 추이")
            fig_rec.update_traces(texttemplate='%{text:.1f}분', textposition='top center')
            st.plotly_chart(fig_rec, use_container_width=True)

    # --- 그래프 2. 시간대별 기상 트렌드 (멀티 셀렉트) ---
    selected_weather = st.multiselect("🌤️ 시각화할 기상 지표 선택", options=["기온 (°C)", "이슬점 온도 (°C)", "시정 (m)", "풍속 (KT)", "강수량 (mm)"], default=["기온 (°C)", "시정 (m)"])
    w_map = {"기온 (°C)": "Avg_Temp", "이슬점 온도 (°C)": "Avg_Dew", "시정 (m)": "Avg_Vis", "풍속 (KT)": "Avg_Wind", "강수량 (mm)": "Avg_Precip"}
    
    if selected_weather:
        fig_w = make_subplots(specs=[[{"secondary_y": True}]])
        colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
        for i, w_name in enumerate(selected_weather):
            col = w_map[w_name]
            is_sec = w_name in ["시정 (m)", "강수량 (mm)"]
            if w_name == "강수량 (mm)":
                fig_w.add_trace(go.Bar(x=w_hour['Hour'], y=w_hour[col], name=w_name, marker_color=colors[i]), secondary_y=is_sec)
            else:
                fig_w.add_trace(go.Scatter(x=w_hour['Hour'], y=w_hour[col], name=w_name, line=dict(color=colors[i])), secondary_y=is_sec)
        fig_w.update_layout(title=f"{sel_date} 시간대별 기상 트렌드", hovermode="x unified")
        st.plotly_chart(fig_w, use_container_width=True)
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
        
        # =========================================================
        # 🌟 [신규 추가] 항공사 그룹별 강설 회복 탄력성 (FSC vs LCC 비교)
        # =========================================================
        st.divider()
        with st.expander("📉 항공사 그룹별 강설 회복 곡선 (FSC vs LCC) 보기 (클릭하여 펼치기)"):
            st.markdown("**대형사(FSC)와 저비용항공사(LCC)** 등 항공사 그룹별로 눈보라 이후 지상이동시간이 얼마나 빨리 정상화되는지 '인프라 회복력'을 직접 비교합니다.")
            
            # 이륙 전 제빙 및 대기 시간이 가장 중요하므로 'Taxi-Out'을 기준으로 그룹핑
            airline_recovery = flights.groupby(['Snow_Status', 'Airline_Group'])['Taxi_Out'].mean().reset_index()
            
            if not airline_recovery.empty:
                fig_a_rec = px.line(
                    airline_recovery, 
                    x='Snow_Status', 
                    y='Taxi_Out', 
                    color='Airline_Group',
                    markers=True,
                    title="항공사 그룹별 강설 영향권에 따른 평균 Taxi-Out 회복 추이"
                )
                fig_a_rec.update_traces(line=dict(width=4), marker=dict(size=10))
                fig_a_rec.update_layout(hovermode="x unified")
                fig_a_rec.update_yaxes(title_text="평균 Taxi-Out 시간 (분)")
                fig_a_rec.update_xaxes(title_text="강설 영향권 (회복 단계)")
                
                st.plotly_chart(fig_a_rec, use_container_width=True)

    
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
