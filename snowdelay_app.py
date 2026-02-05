import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# -----------------------------------------------------------
# [설정] 연도별 파일 이름 매핑
# -----------------------------------------------------------
DATA_FILES = {
    2023: {
        "weather": "AMOS_RKSI_2023.csv",
        "ramp": "2023_RAMP_with_STD_v3.csv",
        "snow": "snow_AMOS_RKSI_2023.csv"
    },
    2024: {
        "weather": "AMOS_RKSI_2024.csv",
        "ramp": "2024_RAMP_with_STD_v3.csv",
        "snow": "snow_AMOS_RKSI_2024.csv"
    },
    2025: {
        "weather": "AMOS_RKSI_2025.csv",
        "ramp": "2025_RAMP_with_STD_v3.csv", 
        "snow": "snow_AMOS_RKSI_2025.csv"
    }
}

# -----------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------
st.set_page_config(page_title="인천공항 운영/기상 대시보드", layout="wide")

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

    # [내부 함수 1] 안전하게 파일 읽기
    def read_csv_safe(filepath):
        encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, encoding=enc, engine='python')
                df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
                
                # 깨진 컬럼명(BOM) 강제 수정
                if isinstance(df.columns[0], str) and 'ate' in df.columns[0] and len(df.columns[0]) > 4:
                      new_cols = list(df.columns)
                      new_cols[0] = 'Date'
                      df.columns = new_cols
                return df
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        raise ValueError(f"파일을 읽을 수 없습니다: {filepath}")

    # [내부 함수 2] 날짜 컬럼 찾기
    def find_date_column(df, filename):
        candidates = ['Date', 'date', 'DATE', '일자', '날짜', 'OpDate']
        for col in df.columns:
            if col in candidates:
                return col
        raise KeyError(f"'{filename}' 파일에서 날짜 컬럼을 찾을 수 없습니다.\n현재 컬럼 목록: {list(df.columns)}")

    # 1. 데이터 불러오기
    try:
        df_weather = read_csv_safe(files['weather'])
        df_ramp = read_csv_safe(files['ramp'])
        df_snow = read_csv_safe(files['snow'])
    except Exception as e:
        st.error(f"파일 로딩 실패 ({year}년): {e}")
        st.stop()

    # --- 기상 데이터 전처리 ---
    df_weather['일시'] = pd.to_datetime(df_weather['일시'])
    df_weather['Month'] = df_weather['일시'].dt.month
    df_weather['Day'] = df_weather['일시'].dt.day
    df_weather['Hour'] = df_weather['일시'].dt.hour
    
    # 상대습도 계산 (Magnus 공식 활용)
    def calculate_rh(row):
        T = row['기온(°C)']
        Td = row['이슬점온도(°C)']
        if pd.isna(T) or pd.isna(Td):
            return None
        
        a = 17.625
        b = 243.04
        
        try:
            es = np.exp((a * T) / (b + T))
            e  = np.exp((a * Td) / (b + Td))
            rh = (e / es) * 100
            return min(100, max(0, rh))
        except:
            return None

    if '기온(°C)' in df_weather.columns and '이슬점온도(°C)' in df_weather.columns:
        df_weather['상대습도(%)'] = df_weather.apply(calculate_rh, axis=1)
    else:
        df_weather['상대습도(%)'] = None

    # --- 눈 데이터 전처리 ---
    df_snow['일시'] = pd.to_datetime(df_snow['일시'])
    df_snow['Month'] = df_snow['일시'].dt.month
    df_snow['Day'] = df_snow['일시'].dt.day
    df_snow['Hour'] = df_snow['일시'].dt.hour
    
    # --- RAMP 데이터 전처리 ---
    date_col_name = find_date_column(df_ramp, files['ramp'])
    df_ramp.rename(columns={date_col_name: 'Date'}, inplace=True)

    df_ramp['Date'] = df_ramp['Date'].astype(str)
    df_ramp['Date_dt'] = pd.to_datetime(df_ramp['Date'], format='%y%m%d', errors='coerce')
    
    # 시간 추출 함수
    def get_hour_from_str(x):
        try:
            return int(str(x).split(':')[0])
        except:
            return None
    
    # 지연 시간 계산 (ATD - STD)
    def calculate_delay_minutes(row):
        try:
            std_h, std_m = map(int, str(row['STD']).split(':'))
            atd_h, atd_m = map(int, str(row['ATD']).split(':'))
            
            std_mins = std_h * 60 + std_m
            atd_mins = atd_h * 60 + atd_m
            
            diff = atd_mins - std_mins
            
            if diff < -720:  
                diff += 1440
            elif diff > 720: 
                diff -= 1440
                
            return diff
        except:
            return None

    # 컬럼 생성
    df_ramp['STD_Hour'] = df_ramp['STD'].apply(get_hour_from_str)
    df_ramp['ATD_Hour'] = df_ramp['ATD'].apply(get_hour_from_str)
    df_ramp['Delay_Min'] = df_ramp.apply(calculate_delay_minutes, axis=1)
    df_ramp['Month'] = df_ramp['Date_dt'].dt.month
    df_ramp['Day'] = df_ramp['Date_dt'].dt.day
    
    return df_weather, df_ramp, df_snow

# 데이터 로드
try:
    df_weather, df_ramp, df_snow = load_data(selected_year)
except Exception as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------------------------------------
# 3. 사이드바: 월/일 선택
# -----------------------------------------------------------
available_months = sorted(df_weather['Month'].unique())
selected_month = st.sidebar.selectbox("월(Month)을 선택하세요", available_months)

available_days = sorted(df_weather[df_weather['Month'] == selected_month]['Day'].unique())
selected_day = st.sidebar.selectbox("일(Day)을 선택하세요", available_days)

# -----------------------------------------------------------
# 4. 데이터 필터링 및 집계
# -----------------------------------------------------------
daily_weather = df_weather[(df_weather['Month'] == selected_month) & (df_weather['Day'] == selected_day)]
daily_snow = df_snow[(df_snow['Month'] == selected_month) & (df_snow['Day'] == selected_day)]
daily_ramp = df_ramp[(df_ramp['Month'] == selected_month) & (df_ramp['Day'] == selected_day)]

# 1. [계획] 시간별 계획된 운항 수 (STD 기준) - 모든 운항 포함
hourly_planned = daily_ramp.groupby('STD_Hour').size().reindex(range(24), fill_value=0).reset_index(name='Planned_Count')

# 2. [실제] 시간별 실제 운항 수 (ATD 기준)
df_actual_base = daily_ramp[daily_ramp['STS'].isin(['DEP', 'DLA'])]

# =========================================================================================
# [옵션 1] ATD 기준 집계 시 STD가 없는 데이터 제외하기
# 아래 주석(#)을 해제하면 STD가 비어있는 행은 실제 운항 수 집계에서 제외됩니다.
# =========================================================================================
df_actual_base = df_actual_base[df_actual_base['STD'].notna() & (df_actual_base['STD'] != '')]
# =========================================================================================

hourly_actual = df_actual_base.groupby('ATD_Hour').size().reindex(range(24), fill_value=0).reset_index(name='Actual_Count')

# 3. 시간별 지연 편수 (DLA) - STD 기준
df_delay_base = daily_ramp[daily_ramp['STS'] == 'DLA']

# =========================================================================================
# [옵션 2] 지연 편수 집계 시 STD가 없는 데이터 제외하기
# 아래 주석(#)을 해제하면 STD가 비어있는 행은 지연 편수 집계에서 제외됩니다.
# =========================================================================================
df_delay_base = df_delay_base[df_delay_base['STD'].notna() & (df_delay_base['STD'] != '')]
# =========================================================================================

hourly_delay_count = df_delay_base.groupby('STD_Hour').size().reindex(range(24), fill_value=0).reset_index(name='Delay_Count')

# 4. 시간별 평균 지연 시간 (분) - STD 기준
hourly_delay_time = daily_ramp.groupby('STD_Hour')['Delay_Min'].mean().reindex(range(24)).reset_index(name='Avg_Delay_Min')

# 5. 시간별 평균 ATD-RAM - STD 기준
hourly_atd_ram = daily_ramp[daily_ramp['ATD-RAM'].notnull()].groupby('STD_Hour')['ATD-RAM'].mean().reindex(range(24)).reset_index(name='Avg_ATD_RAM')

# -----------------------------------------------------------
# 5. 메인 화면: 그래프
# -----------------------------------------------------------
st.header(f"📊 {selected_year}년 {selected_month}월 {selected_day}일 상세 분석")

snow_hours = daily_snow['Hour'].unique()

if len(snow_hours) > 0:
    snow_hours_clean = [int(h) for h in sorted(snow_hours)]
    st.info(f"❄️ 강설 관측 시간대: {snow_hours_clean}시 (그래프 배경이 하늘색으로 표시됩니다)")
else:
    st.success("☀️ 이 날은 강설 기록이 없습니다.")

if not daily_weather.empty:
    # 11개의 서브플롯
    fig = make_subplots(
        rows=11, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "계획된 운항 수 (STD 기준)",   
            "실제 운항 수 (ATD 기준)",     
            "지연(DLA) 편수",              
            "평균 지연 시간 (분)", 
            "평균 ATD-RAM (분)",
            "강수량 (mm)",
            "풍속 (KT)", 
            "시정 (m)", 
            "기온 (°C)", 
            "상대습도 (%)", 
            "현지 기압 (hPa)"
        )
    )

    # 1. 계획된 운항 수
    fig.add_trace(go.Bar(x=hourly_planned['STD_Hour'], y=hourly_planned['Planned_Count'], 
                         name="계획된 운항 수", marker_color='navy'), row=1, col=1)

    # 2. 실제 운항 수
    fig.add_trace(go.Bar(x=hourly_actual['ATD_Hour'], y=hourly_actual['Actual_Count'], 
                         name="실제 운항 수", marker_color='teal'), row=2, col=1)

    # 3. 지연 편수
    fig.add_trace(go.Bar(x=hourly_delay_count['STD_Hour'], y=hourly_delay_count['Delay_Count'], 
                         name="지연 편수", marker_color='red'), row=3, col=1)

    # 4. 평균 지연 시간
    fig.add_trace(go.Scatter(x=hourly_delay_time['STD_Hour'], y=hourly_delay_time['Avg_Delay_Min'], 
                             name="평균 지연 시간", mode='lines+markers', line=dict(color='darkred')), row=4, col=1)

    # 5. ATD-RAM
    fig.add_trace(go.Scatter(x=hourly_atd_ram['STD_Hour'], y=hourly_atd_ram['Avg_ATD_RAM'], 
                             name="평균 ATD-RAM", mode='lines+markers', line=dict(color='purple')), row=5, col=1)
                             
    # 6. 강수량
    precip_data = daily_weather['강수량(mm)'].fillna(0) if '강수량(mm)' in daily_weather.columns else [0]*24
    fig.add_trace(go.Bar(x=daily_weather['Hour'], y=precip_data, 
                         name="강수량", marker_color='cornflowerblue'), row=6, col=1)

    # 7. 풍속
    fig.add_trace(go.Scatter(x=daily_weather['Hour'], y=daily_weather['풍속(KT)'], 
                             name="풍속", line=dict(color='orange')), row=7, col=1)

    # 8. 시정
    fig.add_trace(go.Scatter(x=daily_weather['Hour'], y=daily_weather['시정(m)'], 
                             name="시정", fill='tozeroy', line=dict(color='gray')), row=8, col=1)
                             
    # 9. 기온
    fig.add_trace(go.Scatter(x=daily_weather['Hour'], y=daily_weather['기온(°C)'], 
                             name="기온", line=dict(color='green')), row=9, col=1)

    # 10. 상대습도
    fig.add_trace(go.Scatter(x=daily_weather['Hour'], y=daily_weather['상대습도(%)'], 
                             name="상대습도", fill='tozeroy', line=dict(color='deepskyblue')), row=10, col=1)

    # 11. 현지 기압
    fig.add_trace(go.Scatter(x=daily_weather['Hour'], y=daily_weather['현지기압(hPa)'], 
                             name="기압", line=dict(color='blue')), row=11, col=1)

    # 눈 온 시간대 배경 강조
    for h in snow_hours:
        for row in range(1, 12): 
            fig.add_vrect(
                x0=h-0.5, x1=h+0.5, 
                fillcolor="skyblue", 
                opacity=0.3, 
                layer="below", line_width=0, row=row, col=1
            )

    # 레이아웃 설정
    fig.update_layout(height=2400, showlegend=False, hovermode="x unified")
    
    # x축 설정
    fig.update_xaxes(showticklabels=True, title_text=None) 
    fig.update_xaxes(title_text="시간 (Hour)", row=11, col=1)
    fig.update_xaxes(range=[-0.5, 23.5])

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("기상 데이터가 없습니다.")

# -----------------------------------------------------------
# 6. 하단 데이터 테이블
# -----------------------------------------------------------
with st.expander("📂 원본 데이터 보기"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("운항 상세")
        cols = ['FLT', 'STD', 'STD_Hour', 'ATD', 'ATD_Hour', 'STS', 'Delay_Min', 'ATD-RAM']
        existing_cols = [c for c in cols if c in daily_ramp.columns]
        st.dataframe(daily_ramp[existing_cols])
    with col2:
        st.subheader("시간별 기상 상세")
        weather_cols = ['Hour', '풍속(KT)', '시정(m)', '기온(°C)', '상대습도(%)', '현지기압(hPa)']
        if '강수량(mm)' in daily_weather.columns:
            weather_cols.append('강수량(mm)')
        st.dataframe(daily_weather[weather_cols])

