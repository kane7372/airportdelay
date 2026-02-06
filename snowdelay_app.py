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

    def read_csv_safe(filepath):
        encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, encoding=enc, engine='python')
                df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
                if isinstance(df.columns[0], str) and 'ate' in df.columns[0] and len(df.columns[0]) > 4:
                      new_cols = list(df.columns)
                      new_cols[0] = 'Date'
                      df.columns = new_cols
                return df
            except:
                continue
        raise ValueError(f"파일을 읽을 수 없습니다: {filepath}")

    def find_date_column(df, filename):
        candidates = ['Date', 'date', 'DATE', '일자', '날짜', 'OpDate']
        for col in df.columns:
            if col in candidates:
                return col
        raise KeyError(f"날짜 컬럼 없음: {filename}")

    try:
        df_weather = read_csv_safe(files['weather'])
        df_ramp = read_csv_safe(files['ramp'])
        df_snow = read_csv_safe(files['snow'])
    except Exception as e:
        st.error(f"파일 로딩 실패: {e}")
        st.stop()

    # --- 기상 전처리 ---
    df_weather['일시'] = pd.to_datetime(df_weather['일시'])
    df_weather['Month'] = df_weather['일시'].dt.month
    df_weather['Day'] = df_weather['일시'].dt.day
    df_weather['Hour'] = df_weather['일시'].dt.hour
    
    def calculate_rh(row):
        try:
            T, Td = row['기온(°C)'], row['이슬점온도(°C)']
            if pd.isna(T) or pd.isna(Td): return None
            es = np.exp((17.625 * T) / (243.04 + T))
            e  = np.exp((17.625 * Td) / (243.04 + Td))
            return min(100, max(0, (e/es)*100))
        except: return None

    if '기온(°C)' in df_weather.columns and '이슬점온도(°C)' in df_weather.columns:
        df_weather['상대습도(%)'] = df_weather.apply(calculate_rh, axis=1)
    else:
        df_weather['상대습도(%)'] = None

    # --- 눈 전처리 ---
    df_snow['일시'] = pd.to_datetime(df_snow['일시'])
    df_snow['Month'] = df_snow['일시'].dt.month
    df_snow['Day'] = df_snow['일시'].dt.day
    df_snow['Hour'] = df_snow['일시'].dt.hour
    
    # --- RAMP 전처리 ---
    date_col = find_date_column(df_ramp, files['ramp'])
    df_ramp.rename(columns={date_col: 'Date'}, inplace=True)
    df_ramp['Date_dt'] = pd.to_datetime(df_ramp['Date'].astype(str), format='%y%m%d', errors='coerce')
    
    def get_hour(x):
        try: return int(str(x).split(':')[0])
        except: return None
    
    def calc_delay(row):
        try:
            sh, sm = map(int, str(row['STD']).split(':'))
            ah, am = map(int, str(row['ATD']).split(':'))
            diff = (ah*60+am) - (sh*60+sm)
            if diff < -720: diff += 1440
            elif diff > 720: diff -= 1440
            return diff
        except: return None

    df_ramp['STD_Hour'] = df_ramp['STD'].apply(get_hour)
    df_ramp['ATD_Hour'] = df_ramp['ATD'].apply(get_hour)
    df_ramp['Delay_Min'] = df_ramp.apply(calc_delay, axis=1)
    df_ramp['Month'] = df_ramp['Date_dt'].dt.month
    df_ramp['Day'] = df_ramp['Date_dt'].dt.day
    
    return df_weather, df_ramp, df_snow

try:
    df_weather, df_ramp, df_snow = load_data(selected_year)
except Exception as e:
    st.error(f"오류: {e}")
    st.stop()

# -----------------------------------------------------------
# 3. 사이드바 설정 (날짜 및 옵션)
# -----------------------------------------------------------
avail_months = sorted(df_weather['Month'].unique())
selected_month = st.sidebar.selectbox("월(Month)", avail_months)
avail_days = sorted(df_weather[df_weather['Month'] == selected_month]['Day'].unique())
selected_day = st.sidebar.selectbox("일(Day)", avail_days)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 집계 옵션")
# [옵션 스위치]
exclude_no_std_actual = st.sidebar.checkbox("실제 운항 수에서 계획(STD) 없는 편 제외", value=False)
exclude_no_std_delay = st.sidebar.checkbox("지연 편수에서 계획(STD) 없는 편 제외", value=False)

# -----------------------------------------------------------
# 4. 데이터 필터링 및 집계
# -----------------------------------------------------------
d_weather = df_weather[(df_weather['Month'] == selected_month) & (df_weather['Day'] == selected_day)]
d_snow = df_snow[(df_snow['Month'] == selected_month) & (df_snow['Day'] == selected_day)]
d_ramp = df_ramp[(df_ramp['Month'] == selected_month) & (df_ramp['Day'] == selected_day)]

# (1) 계획된 운항 수
h_planned = d_ramp.groupby('STD_Hour').size().reindex(range(24), fill_value=0).reset_index(name='Planned_Count')

# (2) 실제 운항 수
d_actual_base = d_ramp[d_ramp['STS'].isin(['DEP', 'DLA'])]
if exclude_no_std_actual:
    d_actual_base = d_actual_base[d_actual_base['STD'].notna() & (d_actual_base['STD'] != '')]
h_actual = d_actual_base.groupby('ATD_Hour').size().reindex(range(24), fill_value=0).reset_index(name='Actual_Count')

# (3) 지연 편수
d_delay_base = d_ramp[d_ramp['STS'] == 'DLA']
if exclude_no_std_delay:
    d_delay_base = d_delay_base[d_delay_base['STD'].notna() & (d_delay_base['STD'] != '')]
h_delay_count = d_delay_base.groupby('STD_Hour').size().reindex(range(24), fill_value=0).reset_index(name='Delay_Count')

# (4) 평균 지연/ATD-RAM
h_delay_time = d_ramp.groupby('STD_Hour')['Delay_Min'].mean().reindex(range(24)).reset_index(name='Avg_Delay_Min')
h_atd_ram = d_ramp[d_ramp['ATD-RAM'].notnull()].groupby('STD_Hour')['ATD-RAM'].mean().reindex(range(24)).reset_index(name='Avg_ATD_RAM')

# (5) 강수량 데이터 준비
precip_data = d_weather['강수량(mm)'].fillna(0) if '강수량(mm)' in d_weather.columns else [0]*24

# -----------------------------------------------------------
# 5. 그래프 정의 및 순서 설정 (Drag & Drop 대안)
# -----------------------------------------------------------
# 모든 가능한 그래프의 정의를 딕셔너리로 만듭니다.
GRAPH_CONFIG = {
    "시간당 계획된 운항 수 (STD)": {
        "x": h_planned['STD_Hour'], "y": h_planned['Planned_Count'], "type": "bar", "color": "navy"
    },
    "시간당 실제 운항 수 (ATD)": {
        "x": h_actual['ATD_Hour'], "y": h_actual['Actual_Count'], "type": "bar", "color": "teal"
    },
    "시간당 지연 편수 (DLA)": {
        "x": h_delay_count['STD_Hour'], "y": h_delay_count['Delay_Count'], "type": "bar", "color": "red"
    },
    "시간당 평균 지연 (분)": {
        "x": h_delay_time['STD_Hour'], "y": h_delay_time['Avg_Delay_Min'], "type": "line", "color": "darkred"
    },
    "시간당 평균 지상이동 (분)": {
        "x": h_atd_ram['STD_Hour'], "y": h_atd_ram['Avg_ATD_RAM'], "type": "line", "color": "purple"
    },
    "시간당 강수량 (mm)": {
        "x": d_weather['Hour'], "y": precip_data, "type": "bar", "color": "cornflowerblue"
    },
    "시간당 풍속 (KT)": {
        "x": d_weather['Hour'], "y": d_weather['풍속(KT)'], "type": "line", "color": "orange"
    },
    "시간당 시정 (m)": {
        "x": d_weather['Hour'], "y": d_weather['시정(m)'], "type": "area", "color": "gray"
    },
    "시간당 기온 (°C)": {
        "x": d_weather['Hour'], "y": d_weather['기온(°C)'], "type": "line", "color": "green"
    },
    "시간당 상대습도 (%)": {
        "x": d_weather['Hour'], "y": d_weather['상대습도(%)'], "type": "area", "color": "deepskyblue"
    },
    "시간당 현지 기압 (hPa)": {
        "x": d_weather['Hour'], "y": d_weather['현지기압(hPa)'], "type": "line", "color": "blue"
    }
}

st.sidebar.markdown("---")
st.sidebar.subheader("📊 그래프 순서 및 표시 설정")
st.sidebar.info("아래 목록에서 순서를 바꾸면 그래프 순서가 변경됩니다. 항목을 삭제하면 그래프가 숨겨집니다.")

# 기본 순서 정의
default_order = list(GRAPH_CONFIG.keys())

# 멀티셀렉트로 순서 변경 UI 제공
selected_graphs = st.sidebar.multiselect(
    "그래프 순서 변경 (드래그하여 순서 조정 가능)",
    options=default_order,
    default=default_order
)

# -----------------------------------------------------------
# 6. 메인 화면: 동적 그래프 그리기
# -----------------------------------------------------------
st.header(f"📊 {selected_year}년 {selected_month}월 {selected_day}일 상세 분석")

snow_hours = d_snow['Hour'].unique()
if len(snow_hours) > 0:
    snow_clean = [int(h) for h in sorted(snow_hours)]
    st.info(f"❄️ 강설 관측: {snow_clean}시 (하늘색 배경)")
else:
    st.success("☀️ 강설 없음")

if not d_weather.empty and selected_graphs:
    # 선택된 그래프 개수에 맞춰 서브플롯 생성
    rows_count = len(selected_graphs)
    fig = make_subplots(
        rows=rows_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=selected_graphs
    )

    # 선택된 순서대로 그래프 추가
    for i, graph_name in enumerate(selected_graphs):
        conf = GRAPH_CONFIG[graph_name]
        row_idx = i + 1
        
        if conf['type'] == 'bar':
            fig.add_trace(go.Bar(x=conf['x'], y=conf['y'], name=graph_name, marker_color=conf['color']), row=row_idx, col=1)
        elif conf['type'] == 'line':
            fig.add_trace(go.Scatter(x=conf['x'], y=conf['y'], name=graph_name, mode='lines+markers', line=dict(color=conf['color'])), row=row_idx, col=1)
        elif conf['type'] == 'area':
            fig.add_trace(go.Scatter(x=conf['x'], y=conf['y'], name=graph_name, fill='tozeroy', line=dict(color=conf['color'])), row=row_idx, col=1)

    # 눈 온 시간대 배경 (모든 서브플롯에 적용)
    for h in snow_hours:
        for r in range(1, rows_count + 1):
            fig.add_vrect(
                x0=h-0.5, x1=h+0.5, 
                fillcolor="skyblue", opacity=0.3, 
                layer="below", line_width=0, row=r, col=1
            )

    fig.update_layout(height=200 * rows_count + 200, showlegend=False, hovermode="x unified")
    fig.update_xaxes(showticklabels=True, title_text=None)
    fig.update_xaxes(title_text="시간 (Hour)", row=rows_count, col=1)
    fig.update_xaxes(range=[-0.5, 23.5])

    st.plotly_chart(fig, use_container_width=True)
elif not selected_graphs:
    st.warning("선택된 그래프가 없습니다. 사이드바에서 그래프를 선택해주세요.")
else:
    st.warning("기상 데이터가 없습니다.")

# -----------------------------------------------------------
# 7. 하단 데이터 테이블
# -----------------------------------------------------------
with st.expander("📂 원본 데이터 보기"):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("운항 상세")
        cols = ['FLT', 'STD', 'ATD', 'STS', 'Delay_Min', 'ATD-RAM']
        exist = [c for c in cols if c in d_ramp.columns]
        st.dataframe(d_ramp[exist])
    with c2:
        st.subheader("기상 상세")
        w_cols = ['Hour', '풍속(KT)', '시정(m)', '기온(°C)', '상대습도(%)', '현지기압(hPa)']
        if '강수량(mm)' in d_weather.columns: w_cols.append('강수량(mm)')
        st.dataframe(d_weather[w_cols])
# -----------------------------------------------------------


