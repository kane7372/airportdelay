import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------
# [설정] 연도별 파일 이름 매핑
# 파일명이 다르다면 이 부분을 수정해주세요.
# -----------------------------------------------------------
DATA_FILES = {
    2023: {
        "weather": "AMOS_RKSI_2023.csv",
        "ramp": "RAMP_2023.csv",
        "snow": "snow_AMOS_RKSI_2023.csv"
    },
    2024: {
        "weather": "AMOS_RKSI_2024.csv",
        "ramp": "RAMP_2024.csv",
        "snow": "snow_AMOS_RKSI_2024.csv"
    },
    2025: {
        "weather": "AMOS_RKSI_2025.csv",
        "ramp": "2025_RAMP_with_STD_v3.csv", # 기존 파일명 유지
        "snow": "snow_AMOS_RKSI_2025.csv"
    }
}

# -----------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------
st.set_page_config(page_title="인천공항 운영/기상 대시보드", layout="wide")

# 사이드바에서 연도 선택
st.sidebar.header("📅 조회 옵션")
selected_year = st.sidebar.selectbox("연도(Year)를 선택하세요", [2025, 2024, 2023])

st.title(f"🛫 인천공항 {selected_year}년 운영 및 기상 분석")

# -----------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------
@st.cache_data
def load_data(year):
    files = DATA_FILES.get(year)
    
    if not files:
        return None, None, None

    # 파일 읽기
    df_weather = pd.read_csv(files['weather'])
    df_ramp = pd.read_csv(files['ramp'])
    df_snow = pd.read_csv(files['snow'])

    # --- 기상 데이터 전처리 ---
    df_weather['일시'] = pd.to_datetime(df_weather['일시'])
    df_weather['Month'] = df_weather['일시'].dt.month
    df_weather['Day'] = df_weather['일시'].dt.day
    df_weather['Hour'] = df_weather['일시'].dt.hour
    
    # --- 눈 데이터 전처리 ---
    df_snow['일시'] = pd.to_datetime(df_snow['일시'])
    df_snow['Month'] = df_snow['일시'].dt.month
    df_snow['Day'] = df_snow['일시'].dt.day
    df_snow['Hour'] = df_snow['일시'].dt.hour
    
    # --- RAMP 데이터 전처리 ---
    # 날짜 포맷 처리 (250103 -> 2025-01-03)
    # 연도별로 파일의 날짜 형식이 다를 수 있으니 주의해야 합니다. 
    # 여기서는 6자리(YYMMDD)라고 가정합니다.
    df_ramp['Date'] = df_ramp['Date'].astype(str)
    
    # 날짜 파싱 (에러 방지를 위해 errors='coerce')
    df_ramp['Date_dt'] = pd.to_datetime(df_ramp['Date'], format='%y%m%d', errors='coerce')
    
    # STD에서 시간 추출
    def get_hour(x):
        try:
            return int(str(x).split(':')[0])
        except:
            return None
            
    df_ramp['Hour'] = df_ramp['STD'].apply(get_hour)
    df_ramp['Month'] = df_ramp['Date_dt'].dt.month
    df_ramp['Day'] = df_ramp['Date_dt'].dt.day
    
    return df_weather, df_ramp, df_snow

# 데이터 불러오기 시도
try:
    df_weather, df_ramp, df_snow = load_data(selected_year)
except FileNotFoundError:
    st.error(f"⚠️ {selected_year}년도 데이터 파일을 찾을 수 없습니다.")
    st.info(f"GitHub 저장소에 다음 파일들이 있는지 확인해주세요: {DATA_FILES[selected_year]}")
    st.stop()
except Exception as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------------------------------------
# 3. 사이드바: 월(Month) 및 일(Day) 선택
# -----------------------------------------------------------
# 데이터에 존재하는 월만 추출
available_months = sorted(df_weather['Month'].unique())
selected_month = st.sidebar.selectbox("월(Month)을 선택하세요", available_months)

# 선택된 월에 데이터가 있는 날짜만 추출
available_days = sorted(df_weather[df_weather['Month'] == selected_month]['Day'].unique())
selected_day = st.sidebar.selectbox("일(Day)을 선택하세요", available_days)

# -----------------------------------------------------------
# 4. 데이터 필터링
# -----------------------------------------------------------
# 선택된 날짜의 데이터만 걸러내기
current_date_str = f"{selected_year}-{selected_month}-{selected_day}"

daily_weather = df_weather[(df_weather['Month'] == selected_month) & (df_weather['Day'] == selected_day)]
daily_snow = df_snow[(df_snow['Month'] == selected_month) & (df_snow['Day'] == selected_day)]

# RAMP 데이터 필터링
daily_ramp = df_ramp[(df_ramp['Month'] == selected_month) & (df_ramp['Day'] == selected_day)]

# 지연(DLA) 데이터 집계
hourly_delay = daily_ramp[daily_ramp['STS'] == 'DLA'].groupby('Hour').size().reindex(range(24), fill_value=0).reset_index(name='Delay_Count')

# ATD-RAM 평균 집계
hourly_atd_ram = daily_ramp[daily_ramp['ATD-RAM'].notnull()].groupby('Hour')['ATD-RAM'].mean().reindex(range(24)).reset_index(name='Avg_ATD_RAM')

# -----------------------------------------------------------
# 5. 메인 화면: 복합 그래프
# -----------------------------------------------------------
st.header(f"📊 {selected_year}년 {selected_month}월 {selected_day}일 상세 분석")

# 눈 온 시간대 확인
snow_hours = daily_snow['Hour'].unique()
if len(snow_hours) > 0:
    st.info(f"❄️ 강설 관측 시간대: {sorted(snow_hours)}시 (그래프에 파란 배경으로 표시됩니다)")
else:
    st.success("☀️ 이 날은 강설 기록이 없습니다.")

# 그래프 그리기 (데이터가 비어있지 않은 경우에만)
if not daily_weather.empty:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("지연(DLA) 편수", "평균 ATD-RAM (분)", "풍속 (KT)", "시정 (m)")
    )

    # (1) 지연 건수
    fig.add_trace(go.Bar(x=hourly_delay['Hour'], y=hourly_delay['Delay_Count'], 
                         name="지연 건수", marker_color='red'), row=1, col=1)

    # (2) ATD-RAM
    fig.add_trace(go.Scatter(x=hourly_atd_ram['Hour'], y=hourly_atd_ram['Avg_ATD_RAM'], 
                             name="평균 ATD-RAM", mode='lines+markers', line=dict(color='purple')), row=2, col=1)

    # (3) 풍속
    fig.add_trace(go.Scatter(x=daily_weather['Hour'], y=daily_weather['풍속(KT)'], 
                             name="풍속", line=dict(color='orange')), row=3, col=1)

    # (4) 시정
    fig.add_trace(go.Scatter(x=daily_weather['Hour'], y=daily_weather['시정(m)'], 
                             name="시정", fill='tozeroy', line=dict(color='gray')), row=4, col=1)

    # 눈 온 시간대 배경 강조
    for h in snow_hours:
        fig.add_vrect(x0=h-0.5, x1=h+0.5, fillcolor="blue", opacity=0.1, layer="below", line_width=0)

    fig.update_layout(height=1000, showlegend=False, hovermode="x unified")
    fig.update_xaxes(title_text="시간 (Hour)", range=[-0.5, 23.5], row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("해당 날짜의 기상 데이터가 없습니다.")

# -----------------------------------------------------------
# 6. 하단 데이터 테이블
# -----------------------------------------------------------
with st.expander("📂 선택된 날짜의 데이터 원본 보기"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("항공편 지연 목록")
        st.dataframe(daily_ramp[daily_ramp['STS'] == 'DLA'][['FLT', 'STD', 'ATD', 'DES', 'ATD-RAM']])
    with col2:
        st.subheader("시간별 기상")
        st.dataframe(daily_weather[['Hour', '풍속(KT)', '시정(m)', '기온(°C)', '강수량(mm)']])