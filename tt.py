import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import logging

# Thêm hàm kiểm tra chế độ tối/sáng
def is_dark_mode():
     # Kiểm tra xem có session state cho theme không
    if 'theme' in st.session_state:
        return st.session_state.theme == 'dark'
    
    # Hoặc bạn có thể thêm một checkbox để người dùng chọn
    # return st.sidebar.checkbox("Chế độ tối", value=True)
    
    # Mặc định là chế độ tối
    return True

# Trong các hàm tạo biểu đồ
def plot_prophet_style(forecast_result, df, model_name):
    is_dark = is_dark_mode()
    st.write(f"Debug - Chế độ tối: {is_dark}")  # Dòng debug
    point_color = 'white' if is_dark else 'black'
    text_color = 'white' if is_dark else 'black'
    
    fig = go.Figure()
    
    # Dữ liệu lịch sử dạng chấm
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Close'],
        mode='markers',
        name='Observed data points',
        marker=dict(
            color='white' if is_dark_mode() else 'blue',  # Màu điểm thay đổi theo chế độ
    size=4,
    line=dict(
        width=1, 
        color='#1f77b4' if not is_dark_mode() else '#5fafff'  # Viền xanh đậm hơn trong chế độ tối)  # Thêm viền để nổi bật hơn
        ))
    ))
    
    # Các phần khác giữ nguyên...
    
    fig.update_layout(
        template="plotly_white" if not is_dark else "plotly_dark"
    )
    
    return fig

# Imports cho các mô hình dự báo
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    ExponentialSmoothing = None

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# [GIỮ NGUYÊN PHẦN TechnicalAnalyzer CLASS]
class TechnicalAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.calculate_all_indicators()
    
    def calculate_all_indicators(self):
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['SMA_200'] = self.df['Close'].rolling(window=200).mean()
        self.df['EMA_12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA_26'] = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_stochastic()
        self.calculate_volume_indicators()
        self.identify_support_resistance()
        self.calculate_atr()
        self.calculate_adx()
        self.calculate_cci()
        self.calculate_williams_r()
    
    def calculate_rsi(self, period=14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['RSI_Signal'] = 'Neutral'
        self.df.loc[self.df['RSI'] < 30, 'RSI_Signal'] = 'Oversold'
        self.df.loc[self.df['RSI'] > 70, 'RSI_Signal'] = 'Overbought'
    
    def calculate_macd(self):
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Histogram'] = self.df['MACD'] - self.df['Signal_Line']
        self.df['MACD_Signal'] = 'Neutral'
        self.df.loc[self.df['MACD'] > self.df['Signal_Line'], 'MACD_Signal'] = 'Bullish'
        self.df.loc[self.df['MACD'] < self.df['Signal_Line'], 'MACD_Signal'] = 'Bearish'
    
    def calculate_bollinger_bands(self, window=20, num_std=2):
        sma = self.df['Close'].rolling(window=window).mean()
        std = self.df['Close'].rolling(window=window).std()
        self.df['BB_Upper'] = sma + (std * num_std)
        self.df['BB_Middle'] = sma
        self.df['BB_Lower'] = sma - (std * num_std)
        self.df['BB_%B'] = (self.df['Close'] - self.df['BB_Lower']) / (self.df['BB_Upper'] - self.df['BB_Lower'])
        self.df['BB_Bandwidth'] = (self.df['BB_Upper'] - self.df['BB_Lower']) / self.df['BB_Middle']
    
    def calculate_stochastic(self, k_period=14, d_period=3):
        low_min = self.df['Low'].rolling(window=k_period).min()
        high_max = self.df['High'].rolling(window=k_period).max()
        self.df['Stoch_%K'] = 100 * ((self.df['Close'] - low_min) / (high_max - low_min))
        self.df['Stoch_%D'] = self.df['Stoch_%K'].rolling(window=d_period).mean()
        self.df['Stoch_Signal'] = 'Neutral'
        self.df.loc[self.df['Stoch_%K'] < 20, 'Stoch_Signal'] = 'Oversold'
        self.df.loc[self.df['Stoch_%K'] > 80, 'Stoch_Signal'] = 'Overbought'
    
    def calculate_volume_indicators(self):
        self.df['Volume_SMA_20'] = self.df['Volume'].rolling(window=20).mean()
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()
        self.df['VPT'] = self.df['Volume'] * ((self.df['Close'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1))
        self.df['VPT'] = self.df['VPT'].fillna(0).cumsum()
    
    def identify_support_resistance(self, window=20):
        self.df['Support'] = self.df['Low'].rolling(window=window, center=True).min()
        self.df['Resistance'] = self.df['High'].rolling(window=window, center=True).max()
    
    def calculate_atr(self, period=14):
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.df['ATR'] = true_range.rolling(period).mean()
    
    def calculate_adx(self, period=14):
        high_diff = self.df['High'].diff()
        low_diff = -self.df['Low'].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        if 'ATR' not in self.df.columns:
            self.calculate_atr(period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / self.df['ATR'])
        minus_di = 100 * (minus_dm.rolling(period).mean() / self.df['ATR'])
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        self.df['ADX'] = dx.rolling(period).mean()
        self.df['+DI'] = plus_di
        self.df['-DI'] = minus_di
    
    def calculate_cci(self, period=20):
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        self.df['CCI'] = (tp - sma_tp) / (0.015 * mad)
    
    def calculate_williams_r(self, period=14):
        highest_high = self.df['High'].rolling(period).max()
        lowest_low = self.df['Low'].rolling(period).min()
        self.df['Williams_%R'] = -100 * ((highest_high - self.df['Close']) / (highest_high - lowest_low))
    
    def get_signals_summary(self):
        latest = self.df.iloc[-1]
        signals = {
            'RSI': {'value': latest.get('RSI', 0), 'signal': latest.get('RSI_Signal', 'Neutral'),
                    'interpretation': self.interpret_rsi(latest.get('RSI', 50))},
            'MACD': {'value': latest.get('MACD', 0), 'signal': latest.get('MACD_Signal', 'Neutral'),
                     'interpretation': self.interpret_macd(latest.get('MACD', 0), latest.get('Signal_Line', 0))},
            'Stochastic': {'value': latest.get('Stoch_%K', 0), 'signal': latest.get('Stoch_Signal', 'Neutral'),
                          'interpretation': self.interpret_stochastic(latest.get('Stoch_%K', 50))},
            'Bollinger_Bands': {'position': latest.get('BB_%B', 0.5),
                               'interpretation': self.interpret_bb_position(latest.get('BB_%B', 0.5))},
            'ADX': {'value': latest.get('ADX', 0), 'interpretation': self.interpret_adx(latest.get('ADX', 0))},
            'CCI': {'value': latest.get('CCI', 0), 'interpretation': self.interpret_cci(latest.get('CCI', 0))},
            'Williams_%R': {'value': latest.get('Williams_%R', 0),
                           'interpretation': self.interpret_williams_r(latest.get('Williams_%R', -50))}
        }
        return signals
    
    @staticmethod
    def interpret_rsi(rsi_value):
        if rsi_value < 30: return "Quá bán - Có thể tăng giá"
        elif rsi_value > 70: return "Quá mua - Có thể giảm giá"
        elif 30 <= rsi_value <= 40: return "Gần vùng quá bán"
        elif 60 <= rsi_value <= 70: return "Gần vùng quá mua"
        else: return "Trung tính"
    
    @staticmethod
    def interpret_macd(macd, signal):
        if macd > signal: return "Tín hiệu tăng - MACD trên đường tín hiệu"
        elif macd < signal: return "Tín hiệu giảm - MACD dưới đường tín hiệu"
        else: return "Trung tính - Giao cắt"
    
    @staticmethod
    def interpret_stochastic(stoch_k):
        if stoch_k < 20: return "Quá bán - Cơ hội mua"
        elif stoch_k > 80: return "Quá mua - Cảnh báo bán"
        else: return "Trung tính"
    
    @staticmethod
    def interpret_bb_position(bb_percent):
        if bb_percent > 0.8: return "Gần dải trên - Có thể điều chỉnh"
        elif bb_percent < 0.2: return "Gần dải dưới - Có thể phục hồi"
        else: return "Trong dải - Ổn định"
    
    @staticmethod
    def interpret_adx(adx_value):
        if adx_value < 20: return "Xu hướng yếu"
        elif adx_value < 40: return "Xu hướng trung bình"
        else: return "Xu hướng mạnh"
    
    @staticmethod
    def interpret_cci(cci_value):
        if cci_value > 100: return "Quá mua"
        elif cci_value < -100: return "Quá bán"
        else: return "Trung tính"
    
    @staticmethod
    def interpret_williams_r(wr_value):
        if wr_value > -20: return "Quá mua"
        elif wr_value < -80: return "Quá bán"
        else: return "Trung tính"

# Cấu hình trang
st.set_page_config(page_title="Phân Tích Chứng Khoán", page_icon="📈", layout="wide")

# CSS
st.markdown("""
<style>
/* CSS cơ bản */
.main-header {font-size: 2rem; font-weight: bold; color: #1f77b4; margin-bottom: 1rem;}
.section-header {font-size: 1.5rem; font-weight: bold; margin-top: 2rem; margin-bottom: 1rem;}
.metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}

/* Thay đổi slogan "Thông tin thư viện" */
.sidebar .block-container div:has(h3:contains("Thông tin thư viện")) h3 {
    display: none;
}

.sidebar .block-container div:has(h3:contains("Thông tin thư viện")):before {
    content: "Nắm bắt nhịp đập thị trường bằng góc nhìn vượt thời gian - Nơi ba con người tạo nên tương lai đầu tư";
    font-size: 1rem;
    font-weight: bold;
    color: var(--text-color);
    margin-bottom: 10px;
    display: block;
    line-height: 1.4;
}

/* Thích ứng với chế độ sáng */
:root {
    --text-color: #262730;
    --background-color: white;
    --chart-point-color: black;
    --chart-text-color: black;
    --slider-bg-color: #f0f0f0;
}

/* Thích ứng với chế độ tối */
@media (prefers-color-scheme: dark) {
    :root {
        --text-color: white;
        --background-color: #0e1117;
        --chart-point-color: white;
        --chart-text-color: white;
        --slider-bg-color: #333333;
    }
    
    /* Màu chữ trắng cho tiêu đề trong chế độ tối */
    .main-header, .section-header, p, h1, h2, h3, h4, h5, label, .metric-card {
        color: var(--text-color) !important;
    }
    
    /* Màu nền cho metric card */
    .metric-card {
        background-color: #262730 !important;
    }
    
    /* Màu cho tiêu đề biểu đồ */
    h2, h3, .stSubheader {
        color: var(--text-color) !important;
    }
    
    /* Đảm bảo tiêu đề có emoji hiển thị đúng */
    h2:contains("📈"), h3:contains("📊"), h2:contains("🎯"), h3:contains("📉") {
        color: var(--text-color) !important;
    }
    
    /* Màu cho các expander và phần tử khác */
    .streamlit-expanderHeader, .streamlit-expanderContent {
        color: var(--text-color) !important;
    }
    
    /* Màu cho các checkbox, radio, selectbox */
    .stCheckbox>label, .stRadio>label, .stSelectbox>label {
        color: var(--text-color) !important;
    }
    
    /* Màu cho các metric */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }
    
    /* ĐẶC BIỆT: Đảm bảo các tiêu đề chính có màu trắng trong chế độ tối */
    div[data-testid="stMarkdownContainer"] .main-header {
        color: var(--text-color) !important;
    }
    
    /* Đảm bảo tiêu đề có emoji hiển thị đúng */
    div[data-testid="stMarkdownContainer"] div:contains("📈 Dự báo giá tương lai chuyên nghiệp"),
    div[data-testid="stMarkdownContainer"] div:contains("📊 Phân tích tổng quan cổ phiếu"),
    div[data-testid="stMarkdownContainer"] div:contains("📊 Phân tích chỉ số kỹ thuật nâng cao") {
        color: var(--text-color) !important;
    }
}

/* Màu nút xanh dương (luôn giữ màu này bất kể chế độ sáng/tối) */
.stButton>button {
    background-color: #1f77b4 !important;
    color: white !important;
    border: none !important;
}
.stButton>button:hover {
    background-color: #135a8c !important;
}

/* Chỉnh màu cho thanh kéo (slider) */
/* Phần đã kéo - màu xanh */
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background-color: #1f77b4 !important;
    color: white !important;
}

/* Phần chưa kéo - màu trắng hoặc xám nhạt tùy theo chế độ */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}

/* Track của slider - phần đã kéo */
.stSlider [data-baseweb="slider"] div[role="progressbar"] {
    background-color: #1f77b4 !important;
}

/* Track của slider - phần chưa kéo */
.stSlider [data-baseweb="slider"] div[data-testid="stTrack"] {
    background-color: var(--slider-bg-color) !important;
}

/* Đảm bảo giá trị hiển thị trên thanh kéo có màu phù hợp */
[data-testid="stThumbValue"] {
    color: var(--text-color) !important;
}

/* CSS cho biểu đồ Plotly */
.js-plotly-plot .plotly .modebar {
    color: var(--chart-text-color) !important;
}

/* Đảm bảo điểm dữ liệu (observed data points) hiển thị rõ trong chế độ tối */
.js-plotly-plot .plotly .scatter .points path {
    fill: var(--chart-point-color) !important;
}

/* Đảm bảo text trong biểu đồ hiển thị rõ */
.js-plotly-plot .plotly .gtitle, 
.js-plotly-plot .plotly .xtitle, 
.js-plotly-plot .plotly .ytitle,
.js-plotly-plot .plotly .annotation-text {
    fill: var(--chart-text-color) !important;
}

/* Đảm bảo tiêu đề chính có màu phù hợp */
div[data-testid="stAppViewContainer"] div[data-testid="stHeader"] {
    color: var(--text-color) !important;
}

/* Đảm bảo tiêu đề "Dự báo giá tương lai chuyên nghiệp" có màu phù hợp */
div:contains("Dự báo giá tương lai chuyên nghiệp") {
    color: var(--text-color) !important;
}

/* Đảm bảo tiêu đề "Cấu hình dự báo" có màu phù hợp */
div:contains("Cấu hình dự báo") {
    color: var(--text-color) !important;
}

/* Đặc biệt cho phần thanh trượt có màu đỏ */
.stSlider [data-baseweb="slider"] div[role="progressbar"] {
    background-color: #1f77b4 !important;
}
/* CSS đặc biệt cho thanh trượt trong phần cấu hình dự báo */
[data-testid="stExpander"] .stSlider [data-baseweb="slider"] div[role="progressbar"] {
    background-color: #1f77b4 !important;
}

[data-testid="stExpander"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}

</style>
""", unsafe_allow_html=True)





# Sidebar
with st.sidebar:
    st.markdown("### 📊 Ứng Dụng Phân Tích")
    st.markdown("**Phân Tích Thống Kê dự báo cổ phiếu của 3 cô nàng thư giãn**")
    st.markdown("---")
    menu = st.radio("Chọn chức năng:", ["🏠 Trang chủ", "📈 Dự báo", "📊 Chỉ số kỹ thuật nâng cao"], label_visibility="collapsed")

def clean_data(df):
    if df is None or df.empty: return df
    if isinstance(df.index, pd.DatetimeIndex): df = df.reset_index()
    if 'Date' in df.columns: df = df.drop_duplicates(subset=['Date']).sort_values('Date')
    cols = ['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(symbol, start, end, retry_count=0):
    try:
        symbol = symbol.split(',')[0].strip().upper()
        if isinstance(start, (datetime, pd.Timestamp)): start = start.strftime('%Y-%m-%d')
        if isinstance(end, (datetime, pd.Timestamp)): end = end.strftime('%Y-%m-%d')
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty or len(df) == 0:
            if '.' not in symbol:
                st.info(f"Thử tải {symbol}.VN...")
                df = yf.download(f"{symbol}.VN", start=start, end=end, progress=False, auto_adjust=False)
        if df is None or df.empty:
            if retry_count < 2:
                st.warning(f"Retry {retry_count + 1} cho {symbol}...")
                return load_stock_data(symbol, start, end, retry_count + 1)
            return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df = clean_data(df)
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Thiếu cột {col} trong dữ liệu")
                return None
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu {symbol}: {str(e)}")
        if retry_count < 2: return load_stock_data(symbol, start, end, retry_count + 1)
        return None

def calculate_statistics(df):
    stats = {}
    target_cols = ['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in target_cols:
        if col in df.columns:
            idx_min, idx_max = df[col].idxmin(), df[col].idxmax()
            date_min, date_max = df.loc[idx_min, 'Date'], df.loc[idx_max, 'Date']
            stats[col] = {'Mean': df[col].mean(), 'Std': df[col].std(), 'Min': df[col].min(),
                         'Min Date': date_min.strftime('%Y-%m-%d'), 'Max': df[col].max(),
                         'Max Date': date_max.strftime('%Y-%m-%d'), 'Median': df[col].median()}
    return stats

def calculate_correlation(df):
    numeric_cols = ['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in numeric_cols if col in df.columns]
    return df[available_cols].corr()

# ==================== StockForecaster CLASS - PHẦN ĐÃ SỬA ====================
class StockForecaster:
    def __init__(self, df):
        self.df = df.copy()
        self.data = df['Close'].values
        self.dates = df['Date'].values
    
    def calculate_forecast_errors(self, actual, forecast):
        errors = actual - forecast
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(errors / actual)) * 100
        mpe = np.mean(errors / actual) * 100
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'MPE': mpe}
    
    def naive_forecast(self, steps=30):
        try:
            last_value = self.data[-1]
            forecast_values = np.full(steps, last_value)
            changes = np.diff(self.data)
            std_changes = np.std(changes)
            upper = forecast_values + 1.96 * std_changes * np.sqrt(np.arange(1, steps + 1))
            lower = forecast_values - 1.96 * std_changes * np.sqrt(np.arange(1, steps + 1))
            forecast_dates = pd.date_range(
                start=pd.Timestamp(self.df['Date'].iloc[-1]) + pd.Timedelta(days=1),
                periods=steps, freq='D')
            
            # Tính sai số trên dữ liệu lịch sử
            naive_historical = np.roll(self.data, 1)
            naive_historical[0] = naive_historical[1]  # Xử lý giá trị đầu tiên
            errors = self.calculate_forecast_errors(self.data, naive_historical)
            
            
            return {'values': forecast_values, 'dates': forecast_dates, 'upper': upper,
                   'lower': lower, 'method': 'Naïve', 'last_value': last_value, 'errors': errors,  # Thêm thông tin sai số
            'fitted': np.roll(self.data, 1)}
        except Exception as e:
            st.error(f"Lỗi Naïve forecast: {e}")
            return None
    
    def drift_forecast(self, steps=30):
        try:
            last_value, first_value, n = self.data[-1], self.data[0], len(self.data)
            drift = (last_value - first_value) / (n - 1)
            forecast_values = last_value + drift * np.arange(1, steps + 1)
            residuals = np.diff(self.data) - drift
            std_residuals = np.std(residuals)
            upper = forecast_values + 1.96 * std_residuals * np.sqrt(np.arange(1, steps + 1))
            lower = forecast_values - 1.96 * std_residuals * np.sqrt(np.arange(1, steps + 1))
            forecast_dates = pd.date_range(
                start=pd.Timestamp(self.df['Date'].iloc[-1]) + pd.Timedelta(days=1),
                periods=steps, freq='D')
            return {'values': forecast_values, 'dates': forecast_dates, 'upper': upper,
                   'lower': lower, 'method': 'Drift', 'drift': drift}
        except Exception as e:
            st.error(f"Lỗi Drift forecast: {e}")
            return None
    
    def moving_average_forecast(self, steps=30, windows=[3,6,9,12]):
        forecasts = {}
        for window in windows:
            if len(self.data) >= window:
                try:
                    ma_values = [np.mean(self.data[i-window:i]) for i in range(window, len(self.data)+1)]
                    last_ma = ma_values[-1]
                    forecast_values = np.full(steps, last_ma)
                    actual_values = self.data[window:]
                    ma_values_array = np.array(ma_values)
                    if len(actual_values) != len(ma_values_array):
                        min_len = min(len(actual_values), len(ma_values_array))
                        actual_values, ma_values_array = actual_values[:min_len], ma_values_array[:min_len]
                    errors = actual_values - ma_values_array
                    std_error = np.std(errors)
                    upper, lower = forecast_values + 1.96 * std_error, forecast_values - 1.96 * std_error
                    forecast_dates = pd.date_range(
                        start=pd.Timestamp(self.df['Date'].iloc[-1]) + pd.Timedelta(days=1),
                        periods=steps, freq='D')
                    errors_metrics = self.calculate_forecast_errors(actual_values, ma_values_array)
                    
                    forecasts[f"MA-{window}"] = {'values': forecast_values, 'dates': forecast_dates,
                                                 'upper': upper, 'lower': lower, 'window': window, 'method': f'Moving Average ({window} periods)',
                    'errors': errors_metrics,
                    'fitted': np.concatenate([np.full(window, np.nan), ma_values_array])}
                except Exception as e:
                    st.warning(f"Không thể tính MA-{window}: {e}")
        return forecasts
    
    def weighted_moving_average_forecast(self, steps=30, window=6):
        try:
            if len(self.data) < window: return None
            weights = np.arange(1, window + 1)
            wma_values = [np.sum(weights * self.data[i-window+1:i+1]) / np.sum(weights) 
                         for i in range(window-1, len(self.data))]
            last_wma = wma_values[-1]
            forecast_values = np.full(steps, last_wma)
            actual_values = self.data[window:]
            errors = actual_values - wma_values
            std_error = np.std(errors)
            upper, lower = forecast_values + 1.96 * std_error, forecast_values - 1.96 * std_error
            forecast_dates = pd.date_range(
                start=pd.Timestamp(self.df['Date'].iloc[-1]) + pd.Timedelta(days=1),
                periods=steps, freq='D')
            return {'values': forecast_values, 'dates': forecast_dates, 'upper': upper,
                   'lower': lower, 'method': f'Weighted MA ({window} periods)', 'window': window}
        except Exception as e:
            st.error(f"Lỗi WMA: {e}")
            return None
    
    def simple_exponential_smoothing(self, steps=30, alpha=None, optimize=False):
        """PHẦN ĐÃ SỬA - Alpha chuẩn = 0.1, tối ưu dựa trên dữ liệu"""
        try:
            if not STATSMODELS_AVAILABLE:
                st.warning("Statsmodels không khả dụng")
                return None
            
            # Nếu không tối ưu và không có alpha, dùng 0.1 (chuẩn)
            if not optimize and alpha is None:
                alpha = 0.1
            
            # Nếu tối ưu, tìm alpha tốt nhất DỰA VÀO DỮ LIỆU
            if optimize:
                best_alpha, best_sse = None, float('inf')
                # Mở rộng phạm vi tìm kiếm từ 0.01 đến 0.99
                for test_alpha in np.arange(0.01, 1.0, 0.1):
                    try:
                        model = ExponentialSmoothing(self.data, trend=None, seasonal=None, 
                                                    initialization_method='estimated')
                        fit = model.fit(smoothing_level=test_alpha, optimized=False)
                        sse = np.sum(fit.resid ** 2)
                        if sse < best_sse:
                            best_sse, best_alpha = sse, test_alpha
                    except: continue
                alpha = best_alpha if best_alpha else 0.1
                if alpha > 0.8:
                    st.warning(f"⚠️ Alpha tối ưu cao ({alpha:.3f}): Mô hình nhạy cảm với dữ liệu gần đây")
                else:
                    st.info(f"✅ Alpha tối ưu tìm được: {alpha:.3f}")
            
            # Fit model với alpha đã chọn
            model = ExponentialSmoothing(self.data, trend=None, seasonal=None, 
                                        initialization_method='estimated')
            fit = model.fit(smoothing_level=alpha, optimized=False)
            forecast_values = fit.forecast(steps)
            residuals = fit.resid
            std_residuals = np.std(residuals)
            upper, lower = forecast_values + 1.96 * std_residuals, forecast_values - 1.96 * std_residuals
            forecast_dates = pd.date_range(
                start=pd.Timestamp(self.df['Date'].iloc[-1]) + pd.Timedelta(days=1),
                periods=steps, freq='D')
            errors = self.calculate_forecast_errors(self.data[1:], fit.fittedvalues[1:])
            return {'values': forecast_values, 'dates': forecast_dates, 'upper': upper, 'lower': lower,
                   'alpha': alpha, 'method': 'Simple Exponential Smoothing', 'errors': errors,
                   'fitted': fit.fittedvalues}
        except Exception as e:
            st.error(f"Lỗi Simple ES: {e}")
            return None
    
    def holt_forecast(self, steps=30, optimize=True, alpha=None, beta=None):
        """PHẦN ĐÃ SỬA - Bỏ tham số bounds không hợp lệ"""
        try:
            if not STATSMODELS_AVAILABLE:
                st.warning("Statsmodels không khả dụng")
                return None
            
            model = ExponentialSmoothing(self.data, trend='add', seasonal=None, 
                                        initialization_method='estimated')
            
            if optimize:
                # TỐI ƯU HÓA DỰA VÀO DỮ LIỆU - KHÔNG DÙNG BOUNDS
                best_alpha, best_beta, best_sse = None, None, float('inf')
                for test_alpha in np.arange(0.05, 0.95, 0.1):
                    for test_beta in np.arange(0.05, 0.95, 0.1):
                        try:
                            fit = model.fit(smoothing_level=test_alpha, smoothing_trend=test_beta, 
                                          optimized=False)
                            sse = np.sum(fit.resid ** 2)
                            if sse < best_sse:
                                best_sse, best_alpha, best_beta = sse, test_alpha, test_beta
                        except: continue
                alpha, beta = (best_alpha, best_beta) if best_alpha else (0.1, 0.1)
                st.info(f"✅ Holt - Alpha: {alpha:.3f}, Beta: {beta:.3f}")
            else:
                alpha, beta = alpha or 0.1, beta or 0.1
            
            fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
            forecast_values = fit.forecast(steps)
            residuals = fit.resid
            std_residuals = np.std(residuals)
            upper = forecast_values + 1.96 * std_residuals * np.sqrt(np.arange(1, steps+1))
            lower = forecast_values - 1.96 * std_residuals * np.sqrt(np.arange(1, steps+1))
            forecast_dates = pd.date_range(
                start=pd.Timestamp(self.df['Date'].iloc[-1]) + pd.Timedelta(days=1),
                periods=steps, freq='D')
            errors = self.calculate_forecast_errors(self.data[1:], fit.fittedvalues[1:])
            return {'values': forecast_values, 'dates': forecast_dates, 'upper': upper, 'lower': lower,
                   'alpha': alpha, 'beta': beta, 'method': 'Holt (Double Exponential Smoothing)',
                   'errors': errors, 'fitted': fit.fittedvalues, 'level': fit.level, 'trend': fit.trend}
        except Exception as e:
            st.error(f"Lỗi Holt: {e}")
            return None
    
    def holt_winters_forecast(self, steps=30, seasonal_periods=12, 
                             trend_type='add', seasonal_type='add', optimize=True):
        """
        Mô hình Holt-Winters (Triple Exponential Smoothing)
        """
        try:
            if not STATSMODELS_AVAILABLE:
                st.warning("Statsmodels không khả dụng")
                return None
            
            # Lưu seasonal_periods vào biến local để tránh lỗi
            _seasonal_periods = seasonal_periods
            
            # Kiểm tra dữ liệu đủ dài
            if len(self.data) < 2 * _seasonal_periods:
                # Điều chỉnh chu kỳ mùa vụ nếu dữ liệu quá ngắn
                old_periods = _seasonal_periods
                _seasonal_periods = max(4, len(self.data) // 3)
                st.warning(f"Dữ liệu ngắn, điều chỉnh chu kỳ mùa vụ: {old_periods} → {_seasonal_periods}")
            
            results = {}
            
            # Danh sách các cấu hình cần thử
            configs = []
            
            if optimize:
                # Thử tất cả các kết hợp
                for trend in ['add', 'mul']:
                    for seasonal in ['add', 'mul']:
                        configs.append((trend, seasonal, 'optimized'))
            else:
                # Chỉ dùng cấu hình cho trước
                configs.append((trend_type, seasonal_type, 'standard'))
            
            for trend, seasonal, config_type in configs:
                try:
                    model = ExponentialSmoothing(
                        self.data,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=_seasonal_periods,
                        initialization_method='estimated'
                    )
                    
                    # Fit với giới hạn tham số
                    if optimize:
                        fit = model.fit(
                            optimized=True,
                            
                            use_brute=False
                        )
                    else:
                        fit = model.fit(
                            smoothing_level=0.1,  # alpha
                            smoothing_trend=0.1,  # beta
                            smoothing_seasonal=0.1,  # gamma
                            optimized=False
                        )
                    
                    forecast_values = fit.forecast(steps)
                    
                    # Khoảng tin cậy
                    residuals = fit.resid
                    std_residuals = np.std(residuals)
                    
                    upper = forecast_values + 1.96 * std_residuals
                    lower = forecast_values - 1.96 * std_residuals
                    
                    forecast_dates = pd.date_range(
                        start=pd.Timestamp(self.df['Date'].iloc[-1]) + pd.Timedelta(days=1),
                        periods=steps
                    )
                    
                    # Tính các chỉ số
                    errors = self.calculate_forecast_errors(
                        self.data[_seasonal_periods:],
                        fit.fittedvalues[_seasonal_periods:]
                    )
                    
                    # Lấy tham số
                    alpha = fit.params.get('smoothing_level', None)
                    beta = fit.params.get('smoothing_trend', None)
                    gamma = fit.params.get('smoothing_seasonal', None)
                    
                    method_name = f"Holt-Winters ({config_type.title()})"
                    if config_type == 'optimized':
                        method_name = f"Holt-Winters (Trend:{trend}, Seasonal:{seasonal})"
                    
                    results[method_name] = {
                        'values': forecast_values,
                        'dates': forecast_dates,
                        'upper': upper,
                        'lower': lower,
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'trend_type': trend,
                        'seasonal_type': seasonal,
                        'seasonal_periods': _seasonal_periods,
                        'method': method_name,
                        'errors': errors,
                        'fitted': fit.fittedvalues,
                        'aic': fit.aic,
                        'bic': fit.bic
                    }
                    
                    st.success(f"{method_name} - AIC: {fit.aic:.2f}, α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")
                    
                except Exception as e:
                    st.warning(f"Không thể fit {trend}/{seasonal}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            st.error(f"Lỗi Holt-Winters: {e}")
            return None
    
    def prophet_forecast(self, steps=30):
        """
        Mô hình Facebook Prophet
        """
        try:
            if not PROPHET_AVAILABLE:
                st.warning("Prophet không khả dụng")
                return None
            
            # Chuẩn bị dữ liệu
            prophet_df = self.df[['Date', 'Close']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Đảm bảo kiểu dữ liệu chính xác
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            prophet_df['y'] = prophet_df['y'].astype(float)
            
            # Tạo model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.95
            )
            
            # Thêm tính mùa vụ tháng
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            
            # Fit model
            model.fit(prophet_df)
            
            # Tạo future dataframe
            future = model.make_future_dataframe(periods=steps)
            
            # Dự báo
            forecast = model.predict(future)
            
            # Lấy phần dự báo tương lai
            future_forecast = forecast.tail(steps)
            
            # Tính các chỉ số từ dữ liệu lịch sử
            historical_forecast = forecast.head(len(self.data))
            errors = self.calculate_forecast_errors(
                self.data,
                historical_forecast['yhat'].values
            )
            
            return {
                'values': future_forecast['yhat'].values,
                'dates': pd.to_datetime(future_forecast['ds'].values),
                'upper': future_forecast['yhat_upper'].values,
                'lower': future_forecast['yhat_lower'].values,
                'method': 'Facebook Prophet',
                'model': model,
                'forecast_full': forecast,
                'errors': errors,
                'trend': future_forecast['trend'].values,
                'seasonal': future_forecast['yearly'].values if 'yearly' in future_forecast.columns else None
            }
            
        except Exception as e:
            st.error(f"Lỗi Prophet: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None

def display_forecast_metrics(forecast_result, model_name):
    """Hiển thị các chỉ số đo độ lệch của mô hình"""
    if forecast_result and 'errors' in forecast_result:
        errors = forecast_result['errors']
        
        st.markdown(f"#### 📊 Chỉ số đo độ lệch - {model_name}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("MAE", f"{errors['MAE']:.4f}", 
                     help="Mean Absolute Error - Sai số tuyệt đối trung bình")
        
        with col2:
            st.metric("MSE", f"{errors['MSE']:.4f}",
                     help="Mean Squared Error - Sai số bình phương trung bình")
        
        with col3:
            st.metric("RMSE", f"{errors['RMSE']:.4f}",
                     help="Root Mean Squared Error - Căn bậc hai của MSE")
        
        with col4:
            st.metric("MAPE", f"{errors['MAPE']:.2f}%",
                     help="Mean Absolute Percentage Error - Sai số phần trăm tuyệt đối")
        
        with col5:
            st.metric("MPE", f"{errors['MPE']:.2f}%",
                     help="Mean Percentage Error - Sai số phần trăm trung bình")
        
        # Đánh giá chất lượng dự báo
        if errors['MAPE'] < 10:
            quality = "🟢 Rất tốt"
        elif errors['MAPE'] < 20:
            quality = "🟡 Tốt"
        elif errors['MAPE'] < 50:
            quality = "🟠 Chấp nhận được"
        else:
            quality = "🔴 Kém"
        
        st.markdown(f"**Chất lượng dự báo:** {quality}")

# ==================== TRANG CHỦ ====================
if menu == "🏠 Trang chủ":
    st.markdown('<div class="main-header" style="color: white;">📊 Phân tích tổng quan cổ phiếu</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        stock_symbol = st.text_input("Mã chứng khoán", value="COP", help="VD: COP, CVX, SLB")
    
    with col2:
        start_date = st.date_input("Ngày bắt đầu", value=datetime(2019, 12, 14))
    
    with col3:
        end_date = st.date_input("Ngày kết thúc", value=datetime.now())
    
    if st.button("🔍 Phân tích", type="primary"):
        st.cache_data.clear()
        with st.spinner("Đang tải dữ liệu..."):
            df = load_stock_data(stock_symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                st.success(f"✅ Đã tải {len(df)} bản ghi dữ liệu")
                
                # THỐNG KÊ TỔNG QUAN
                st.markdown("### 📊 Thống kê tổng quan")
                
                current_price = df['Close'].iloc[-1]
                previous_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                price_change = current_price - previous_price
                price_change_pct = (price_change / previous_price * 100) if previous_price != 0 else 0
                lowest_price = df['Close'].min()
                highest_price = df['Close'].max()
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        label="Giá hiện tại",
                        value=f"${current_price:.2f}",
                        delta=f"{price_change:.2f}"
                    )
                
                with metric_col2:
                    st.metric(
                        label="Thay đổi",
                        value=f"{price_change_pct:.2f}%",
                        delta=f"{price_change_pct:.2f}%"
                    )
                
                with metric_col3:
                    st.metric(
                        label="Thấp nhất",
                        value=f"${lowest_price:.2f}"
                    )
                
                with metric_col4:
                    st.metric(
                        label="Cao nhất",
                        value=f"${highest_price:.2f}"
                    )
                
                # Thông số đầu vào
                st.markdown('<div class="section-header">📋 Thông số đầu vào</div>', unsafe_allow_html=True)
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Mã chứng khoán", stock_symbol)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with info_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Ngày bắt đầu", start_date.strftime("%Y/%m/%d"))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with info_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Ngày kết thúc", end_date.strftime("%Y/%m/%d"))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Dữ liệu nguồn
                st.markdown('<div class="section-header">📊 Dữ liệu nguồn</div>', unsafe_allow_html=True)
                st.dataframe(df.tail(10), use_container_width=True)
                
                # Các tham số thống kê
                st.markdown('<div class="section-header">📈 Phân tích các tham số thống kê</div>', unsafe_allow_html=True)
                stats = calculate_statistics(df)
                
                stats_df = pd.DataFrame(stats).T
                stats_df = stats_df.round(2)
                st.dataframe(stats_df, use_container_width=True)
                
                # Ma trận tương quan
                st.markdown('<div class="section-header">🔗 Ma trận tương quan các biến</div>', unsafe_allow_html=True)
                corr_matrix = calculate_correlation(df)
                
                st.dataframe(
                    corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1),
                    use_container_width=True
                )
                
                # Biểu đồ giá
                st.markdown('<div class="section-header">📉 Biểu đồ biến động giá và khối lượng</div>', unsafe_allow_html=True)
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3]
                )
                
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='blue')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['Open'], name='Open', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='rgba(231, 76, 60, 0.85)', opacity=0.95,
                           marker_line_width=1.2, marker_line_color='rgba(192, 57, 43, 1)', width=86400000 * 0.65),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=600,
                    title_text="Biến động giá và khối lượng giao dịch",
                    showlegend=True
                )
                fig.update_xaxes(title_text="Thời gian", row=2, col=1)
                fig.update_yaxes(title_text="Giá (USD)", row=1, col=1)
                fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("❌ Không thể tải dữ liệu. Vui lòng kiểm tra mã chứng khoán!")

# ==================== TRANG DỰ BÁO ====================
elif menu == "📈 Dự báo":
    st.markdown('<div class="main-header" style="color: white;">📈 Dự báo giá tương lai chuyên nghiệp</div>', unsafe_allow_html=True)
    
    # PHẦN CẤU HÌNH DỰ BÁO
    with st.expander("⚙️ Cấu hình dự báo", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            symbol = st.text_input("Mã chứng khoán", value="COP", key="forecast_symbol")
        
        with col2:
            forecast_days = st.slider("Chu kì dự báo", 7, 730, 180, 
                                      help="Có thể dự báo từ 1 tuần đến 2 năm")
        
        with col3:
            confidence_level = st.slider("Độ tin cậy (%)", 80, 99, 95,
                                        help="Mức độ tin cậy của khoảng dự báo")
        
        with col4:
            history_display_days = st.slider("Lịch sử hiển thị", 30, 365, 90,
                                            help="Số ngày lịch sử hiển thị trên biểu đồ")
        with col5:
            forecast_timeframe = st.selectbox(
                "Khung thời gian dự báo",
                ["Ngày", "Tuần", "Tháng"],
                index=0,
                help="Chọn đơn vị thời gian cho dự báo"
                )

    # Chuyển đổi số ngày dự báo dựa trên khung thời gian
        if forecast_timeframe == "Tuần":
            actual_forecast_days = forecast_days * 7
        elif forecast_timeframe == "Tháng":
            actual_forecast_days = forecast_days * 30
        else:  # Ngày
            actual_forecast_days = forecast_days
    # CHỌN MÔ HÌNH DỰ BÁO
    
    st.markdown("### 🎯 Cấu hình dự báo")

    # Thiết lập mặc định cho tất cả các mô hình
    model_options = ["Moving Average", "Exponential Smoothing", "Holt", "Holt-Winters", "Prophet"]
    model_config = {
        'MA': {
           'windows': [3, 6, 9, 12, 24],
           'use_wma': False,
           'use_naive': True,
           'use_drift': True
           },
        'ES': {
        'alpha': None,
        'optimize': True
        },
         'Holt': {
        'optimize': True,
        'alpha': None,
        'beta': None
    },
        'HW': {
        'seasonal_periods': 12,
        'optimize': True,
        'trend_type': 'add',
        'seasonal_type': 'add'
    },
        'Prophet': {
        'include_history': True
    }
    }

    # Chọn kiểu biểu đồ
    viz_style = st.selectbox(
        "Chọn kiểu biểu đồ:",
        ["Biểu đồ Prophet", "Biểu đồ Holt-Winters (Tối ưu)", "Biểu đồ Holt-Winters (Tiêu chuẩn)", 
         "Biểu đồ Holt (Tham số cố định)", "Biểu đồ Holt (Tham số tối ưu)", 
         "Biểu đồ SES (Alpha tối ưu)", "Biểu đồ SES (Alpha cố định)", "Biểu đồ Moving Average"],
        index=0
    )

    st.info("💡 Chọn kiểu biểu đồ phù hợp với mô hình dự báo. Kết quả sẽ được hiển thị theo kiểu biểu đồ đã chọn.")
    
    # CHẠY DỰ BÁO
    if st.button("🚀 Chạy phân tích và dự báo", type="primary", use_container_width=True):
        with st.spinner("⏳ Đang tải dữ liệu và tính toán dự báo..."):
            # Tải dữ liệu
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*5)  # Tăng lên 5 năm để có đủ dữ liệu
            
            df = load_stock_data(symbol, start_date, end_date)
            
            if df is not None and len(df) > 30:
                st.success(f"✅ Đã tải {len(df)} ngày dữ liệu cho {symbol}")
                
                # Thông tin tổng quan
                st.markdown("## 📊 Thông tin tổng quan")
                
                current_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0
                
                # Tính volatility
                returns = df['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                
                # Tính các chỉ số thống kê
                price_min = df['Close'].tail(90).min()
                price_max = df['Close'].tail(90).max()
                price_avg = df['Close'].tail(90).mean()
                
                col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)
                
                with col_metric1:
                    st.metric(
                        "Giá hiện tại", 
                        f"${current_price:.2f}",
                        f"{price_change_pct:+.2f}%"
                    )
                
                with col_metric2:
                    st.metric("Volatility (1 năm)", f"{volatility:.2f}%")
                
                with col_metric3:
                    st.metric("Giá TB (90 ngày)", f"${price_avg:.2f}")
                
                with col_metric4:
                    st.metric("Cao nhất (90 ngày)", f"${price_max:.2f}")
                
                with col_metric5:
                    st.metric("Thấp nhất (90 ngày)", f"${price_min:.2f}")
                
                # Chạy các mô hình dự báo
                forecaster = StockForecaster(df)
                all_forecasts = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_steps = len(model_options)
                current_step = 0
                
                # Chạy Moving Average và Random Walk
                if "Moving Average" in model_options and 'MA' in model_config:
                    status_text.text("🔄 Đang chạy Moving Average...")
                    
                    # Naïve
                    if model_config['MA'].get('use_naive', False):
                        naive_result = forecaster.naive_forecast(actual_forecast_days)
                        if naive_result:
                            all_forecasts['Naïve (Random Walk)'] = naive_result
                    
                    # Drift
                    if model_config['MA'].get('use_drift', False):
                        drift_result = forecaster.drift_forecast(actual_forecast_days)
                        if drift_result:
                            all_forecasts['Drift (Random Walk)'] = drift_result
                    
                    # Moving Average
                    ma_forecasts = forecaster.moving_average_forecast(
                        actual_forecast_days, 
                        model_config['MA']['windows']
                    )
                    all_forecasts.update(ma_forecasts)
                    
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                # Chạy Exponential Smoothing
                if "Exponential Smoothing" in model_options and 'ES' in model_config:
                    status_text.text("🔄 Đang chạy Exponential Smoothing...")
                    es_result = forecaster.simple_exponential_smoothing(
                        actual_forecast_days,
                        alpha=model_config['ES']['alpha'],
                        optimize=model_config['ES']['optimize']
                    )
                    if es_result:
                        all_forecasts['Simple Exponential Smoothing'] = es_result
                    
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                # Chạy Holt
                if "Holt" in model_options and 'Holt' in model_config:
                    status_text.text("🔄 Đang chạy Holt (Double ES)...")
                    holt_result = forecaster.holt_forecast(
                        actual_forecast_days,
                        optimize=model_config['Holt']['optimize'],
                        alpha=model_config['Holt']['alpha'],
                        beta=model_config['Holt']['beta']
                    )
                    if holt_result:
                        all_forecasts['Holt (Double ES)'] = holt_result
                    
                                        # Thêm phiên bản Holt với tham số cố định
                    holt_fixed_result = forecaster.holt_forecast(
                        actual_forecast_days,
                        optimize=False,
                        alpha=0.1,
                        beta=0.2
                    )
                    if holt_fixed_result:
                        all_forecasts['Holt Fixed (α=0.10, β=0.20)'] = holt_fixed_result
                    
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                # Chạy Holt-Winters
                if "Holt-Winters" in model_options and 'HW' in model_config:
                    status_text.text("🔄 Đang chạy Holt-Winters (Triple ES)...")
                    hw_results = forecaster.holt_winters_forecast(
                        actual_forecast_days,
                        seasonal_periods=model_config['HW']['seasonal_periods'],
                        trend_type=model_config['HW']['trend_type'],
                        seasonal_type=model_config['HW']['seasonal_type'],
                        optimize=model_config['HW']['optimize']
                    )
                    if hw_results:
                        all_forecasts.update(hw_results)
                    
                    # Thêm phiên bản Holt-Winters với tham số tiêu chuẩn
                    hw_standard_results = forecaster.holt_winters_forecast(
                        actual_forecast_days,
                        seasonal_periods=model_config['HW']['seasonal_periods'],
                        trend_type=model_config['HW']['trend_type'],
                        seasonal_type=model_config['HW']['seasonal_type'],
                        optimize=False
                    )
                    if hw_standard_results:
                        # Đổi tên để phân biệt
                        renamed_results = {}
                        for key, value in hw_standard_results.items():
                            new_key = key.replace("(Standard)", "(Hệ số tiêu chuẩn)")
                            renamed_results[new_key] = value
                        all_forecasts.update(renamed_results)
                    
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                # Chạy Prophet
                if "Prophet" in model_options and 'Prophet' in model_config:
                    status_text.text("🔄 Đang chạy Prophet...")
                    prophet_result = forecaster.prophet_forecast(actual_forecast_days)
                    if prophet_result:
                        all_forecasts['Facebook Prophet'] = prophet_result
                    
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Hoàn thành!")
                
                if all_forecasts:
                    # Tạo các hàm vẽ biểu đồ theo kiểu mẫu
                    
                    # 1. Hàm vẽ biểu đồ kiểu Prophet
                    def plot_prophet_style(forecast_result, df, model_name):
                        is_dark = is_dark_mode()
                        point_color = 'white' if is_dark else 'black'
                        text_color = 'white' if is_dark else 'black'  # Màu chữ thay đổi theo chế độ
                        fig = go.Figure()
                        
                        # Thêm đường giá gốc (đường liên tục)
                        fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Close'],
        mode='lines',
        name='Giá thực tế',
        line=dict(color='blue', width=2)
    ))
                        # Dữ liệu lịch sử dạng chấm đen
                        fig.add_trace(go.Scatter(
                            x=df['Date'], 
                            y=df['Close'],
                            mode='markers',
                            name='Observed data points',
                            marker=dict(
                                color='white' if is_dark_mode() else 'blue',  # Màu điểm thay đổi theo chế độ
                                size=4,
                                line=dict(
                                    width=1, 
                                    color='#1f77b4' if not is_dark_mode() else '#5fafff' ) # Viền xanh đậm hơn trong chế độ tối
        )
    ))
                        
                        # Đường dự báo màu xanh
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'], 
                            y=forecast_result['values'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color="#1f77b4", width=2)
                        ))
                        
                        # Khoảng tin cậy
                        if 'upper' in forecast_result and 'lower' in forecast_result:
                            fig.add_trace(go.Scatter(
                                x=forecast_result['dates'], 
                                y=forecast_result['upper'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_result['dates'], 
                                y=forecast_result['lower'],
                                mode='lines',
                                fill='tonexty',
                                fillcolor='rgba(173, 216, 230, 0.3)',
                                line=dict(width=0),
                                name='Uncertainty interval'
                            ))
                        
                        # Đường dọc đánh dấu bắt đầu dự báo
                        last_date = df['Date'].iloc[-1]
                        fig.add_shape(
                            type="line",
                            x0=last_date,
                            x1=last_date,
                            y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#1f77b4", width=1, dash="dash")
    )
                         # Add annotation separately
                        fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Start of Forecast",
        showarrow=False,
        yshift=10,
        font=dict(color="#1f77b4") 
    )
                      
                        # Chú thích cho giá trị cuối cùng và dự báo cuối
                        
                        fig.add_annotation(
                            x=last_date,
                            y=df['Close'].max(),
                            xref="x",
                            yref="y",
                            text="Start of Forecast",
                            showarrow=True,
                            arrowhead=1,
                            ax=40,
                            ay=-40,
                            font=dict(color='white' if "plotly_dark" in fig.layout.template else 'black')
                        )
                        
                        end_forecast = float(forecast_result['values'][-1])
                         
                        fig.add_annotation(
                            x=forecast_result['dates'][-1],
                            y=end_forecast,
                            text=f"End Forecast: {end_forecast:.2f}",
                            showarrow=True,
                            arrowhead=1,
                            ax=-40,
                            ay=-40,
                            font=dict(color='white' if is_dark_mode() else 'black')
                        )
                        
                        fig.update_layout(
                            title="Time Series Forecast with Prophet",
        xaxis_title="Date",
        yaxis_title="Adjusted Close Price",
        legend_title="Legend",
        height=600,
        template="plotly_white" if not is_dark else "plotly_dark"
    )
                        return fig 
                       
                    
                    # 2. Hàm vẽ biểu đồ kiểu Holt-Winters (Tối ưu)
                    def plot_holt_winters_optimized(forecast_result, df, model_name):
                        is_dark = is_dark_mode()
                        template="plotly_white" if not is_dark else "plotly_dark"


                        fig = go.Figure()
                        
                        # Chia dữ liệu thành train/test
                        train_size = int(len(df) * 0.8)
                        train_data = df.iloc[:train_size]
                        test_data = df.iloc[train_size:]
                        
                        # Dữ liệu huấn luyện
                        fig.add_trace(go.Scatter(
                            x=train_data['Date'], 
                            y=train_data['Close'],
                            mode='lines',
                            name='Dữ liệu huấn luyện',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Dữ liệu kiểm tra
                        fig.add_trace(go.Scatter(
                            x=test_data['Date'], 
                            y=test_data['Close'],
                            mode='lines',
                            name='Dữ liệu kiểm tra',
                            line=dict(color='orange', width=2)
                        ))
                        
                        # Dự báo tối ưu
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'], 
                            y=forecast_result['values'],
                            mode='lines',
                            name='Dự báo tối ưu',
                            line=dict(color='green', width=2)
                        ))
                        
                        # Dự báo tối ưu 180 ngày tương lai
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'], 
                            y=forecast_result['values'],
                            mode='lines',
                            name='Dự báo tối ưu 180 ngày tương lai',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Dự báo Holt-Winters (Hệ số tối ưu)",
                            xaxis_title="Ngày",
                            yaxis_title="Giá",
                            legend_title="Legend",
                            height=600,
                            template="plotly_white" if not is_dark else "plotly_dark"

                        )
                        
                        return fig
                    
                    # 3. Hàm vẽ biểu đồ kiểu Holt-Winters (Tiêu chuẩn)
                    def plot_holt_winters_standard(forecast_result, df, model_name):
                        is_dark = is_dark_mode()

                        fig = go.Figure()
                        
                        # Chia dữ liệu thành train/test
                        train_size = int(len(df) * 0.8)
                        train_data = df.iloc[:train_size]
                        test_data = df.iloc[train_size:]
                        
                        # Dữ liệu huấn luyện
                        fig.add_trace(go.Scatter(
                            x=train_data['Date'], 
                            y=train_data['Close'],
                            mode='lines',
                            name='Dữ liệu huấn luyện',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Dữ liệu kiểm tra
                        fig.add_trace(go.Scatter(
                            x=test_data['Date'], 
                            y=test_data['Close'],
                            mode='lines',
                            name='Dữ liệu kiểm tra',
                            line=dict(color='orange', width=2)
                        ))
                        
                        # Dự báo hệ số tiêu chuẩn
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'], 
                            y=forecast_result['values'],
                            mode='lines',
                            name='Dự báo hệ số tiêu chuẩn',
                            line=dict(color='green', width=2)
                        ))
                        
                        # Dự báo hệ số tiêu chuẩn 180 ngày tương lai
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'], 
                            y=forecast_result['values'],
                            mode='lines',
                            name='Dự báo hệ số tiêu chuẩn 180 ngày tương lai',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Thêm thông tin RMSE
                        if 'errors' in forecast_result:
                            rmse_standard = forecast_result['errors']['RMSE']
                            
                            # Tìm RMSE của mô hình tối ưu (nếu có)
                            rmse_optimized = None
                            for name, result in all_forecasts.items():
                                if 'Holt-Winters' in name and 'optimized' in name.lower() and 'errors' in result:
                                    rmse_optimized = result['errors']['RMSE']
                                    break
                            
                            if rmse_optimized:
                                fig.add_annotation(
                                    x=0.05,
                                    y=0.05,
                                    xref="paper",
                                    yref="paper",
                                    text=f"RMSE cho mô hình hệ số tiêu chuẩn: {rmse_standard:.4f}<br>RMSE cho mô hình tối ưu: {rmse_optimized:.4f}",
                                    showarrow=False,
                                    align="left",
                                    bgcolor="white",
                                    bordercolor="black",
                                    borderwidth=1
                                )
                        
                        fig.update_layout(
                            title="Dự báo Holt-Winters (Hệ số tiêu chuẩn)",
                            xaxis_title="Ngày",
                            yaxis_title="Giá",
                            legend_title="Legend",
                            height=600,
                            template="plotly_white" if not is_dark else "plotly_dark"

                        )
                        
                        return fig
                    
                    # 4. Hàm vẽ biểu đồ kiểu Holt (Tham số cố định)
                    def plot_holt_fixed(forecast_result, df, model_name):
                        is_dark = is_dark_mode()

                        fig = go.Figure()
                        
                        # Dữ liệu gốc
                        fig.add_trace(go.Scatter(
                            x=df['Date'], 
                            y=df['Close'],
                            mode='lines',
                            name='Data COP',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Chia dữ liệu thành train/test
                        train_size = int(len(df) * 0.8)
                        train_data = df.iloc[:train_size]
                        test_data = df.iloc[train_size:]
                        
                        # Train COP
                        fig.add_trace(go.Scatter(
                            x=train_data['Date'], 
                            y=train_data['Close'],
                            mode='lines',
                            name='Train COP',
                            line=dict(color='blue', width=2),
                            showlegend=False
                        ))
                        
                        # Test COP
                        fig.add_trace(go.Scatter(
                            x=test_data['Date'], 
                            y=test_data['Close'],
                            mode='lines',
                            name='Test COP',
                            line=dict(color="#ff001e", width=2)
                        ))
                        
                        # HOLT COP fixed
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'], 
                            y=forecast_result['values'],
                            mode='lines',
                            name='HOLT COP fixed (α=0.10, β=0.20)',
                            line=dict(color='green', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Holt Forecast COP (Fixed Params)",
                            height=600,
                            template="plotly_white" if not is_dark else "plotly_dark"

                        )
                        
                        return fig
                    
                    # 5. Hàm vẽ biểu đồ kiểu Holt (Tham số tối ưu)
                    def plot_holt_optimized(forecast_result, df, model_name):
                        is_dark = is_dark_mode()

                        fig = go.Figure()
                        
                        # Dữ liệu gốc COP
                        fig.add_trace(go.Scatter(
                            x=df['Date'], 
                            y=df['Close'],
                            mode='lines',
                            name='Dữ liệu gốc COP',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Chia dữ liệu thành train/test
                        train_size = int(len(df) * 0.8)
                        train_data = df.iloc[:train_size]
                        test_data = df.iloc[train_size:]
                        
                        # Dữ liệu huấn luyện COP
                        fig.add_trace(go.Scatter(
                            x=train_data['Date'], 
                            y=train_data['Close'],
                            mode='lines',
                            name='Dữ liệu huấn luyện COP',
                            line=dict(color='blue', width=2),
                            showlegend=False
                        ))
                        
                        # Dữ liệu kiểm tra COP
                        fig.add_trace(go.Scatter(
                            x=test_data['Date'], 
                            y=test_data['Close'],
                            mode='lines',
                            name='Dữ liệu kiểm tra COP',
                            line=dict(color="#ff001e", width=2)
                        ))
                        
                        # Lấy giá trị alpha và beta từ kết quả
                        alpha = forecast_result.get('alpha', 0.20)
                        beta = forecast_result.get('beta', 0.10)
                        
                        # Dự báo HOLT COP tối ưu
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'], 
                            y=forecast_result['values'],
                            mode='lines',
                            name=f'Dự báo HOLT COP (Grid Optimized Alpha={alpha:.2f}, Beta={beta:.2f})',
                            line=dict(color='orange', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Dự báo Mô hình HOLT COP (Tham số tối ưu từ Grid Search)",
                            xaxis_title="Ngày (Date)",
                            yaxis_title="Đơn vị ($)",
                            height=600,
                            template="plotly_white" if not is_dark else "plotly_dark"

                        )
                        
                        return fig
                    
                    # 6. Hàm vẽ biểu đồ kiểu SES (Alpha tối ưu)
                    def plot_ses_optimized(forecast_result, df, model_name):
                        is_dark = is_dark_mode()

                        fig = go.Figure()
                        
                        # Giá đóng cửa gốc
                        fig.add_trace(go.Scatter(
                            x=df['Date'], 
                            y=df['Close'],
                            mode='lines',
                            name='Original Adj Close',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Lấy alpha từ kết quả
                        alpha = forecast_result.get('alpha', 1.0)
                        
                        # SES với alpha tối ưu
                        if 'fitted' in forecast_result:
                            # Thêm fitted values cho dữ liệu lịch sử
                            fig.add_trace(go.Scatter(
                                x=df['Date'], 
                                y=forecast_result['fitted'],
                                mode='lines',
                                name=f'SES Optimized Alpha ({alpha:.4f})',
                                line=dict(color='green', width=2, dash='dash')
                            ))
                        
                        # Dự báo tương lai
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'], 
                            y=forecast_result['values'],
                            mode='lines',
                            name=f'Future SES Optimized Alpha',
                            line=dict(color='green', width=2, dash='dash'),
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            title=f"Comparison: Original Adj Close vs. SES Optimized Alpha ({alpha:.4f})",
                            xaxis_title="Date",
                            yaxis_title="Adj Close Value",
                            height=600,
                            template="plotly_white" if not is_dark else "plotly_dark"

                        )
                        
                        return fig
                    
                    # 7. Hàm vẽ biểu đồ kiểu SES (Alpha cố định)
                    def plot_ses_fixed(forecast_result, df, model_name):
                        is_dark = is_dark_mode()

                        fig = go.Figure()
                        
                        # Giá đóng cửa gốc
                        fig.add_trace(go.Scatter(
                            x=df['Date'], 
                            y=df['Close'],
                            mode='lines',
                            name='Original Adj Close',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # SES với alpha cố định
                        if 'fitted' in forecast_result:
                            # Thêm fitted values cho dữ liệu lịch sử
                            fig.add_trace(go.Scatter(
                                x=df['Date'], 
                                y=forecast_result['fitted'],
                                mode='lines',
                                name='SES Fixed Alpha (0.1)',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                        
                        # Dự báo tương lai
                        fig.add_trace(go.Scatter(
                            x=forecast_result['dates'], 
                            y=forecast_result['values'],
                            mode='lines',
                            name='Future SES Fixed Alpha',
                            line=dict(color='red', width=2, dash='dash'),
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            title="Comparison: Original Adj Close vs. SES Fixed Alpha (0.1)",
                            xaxis_title="Date",
                            yaxis_title="Adj Close Value",
                            height=600,
                            template="plotly_white" if not is_dark else "plotly_dark"

                        )
                        
                        return fig
                    
                    # 8. Hàm vẽ biểu đồ kiểu Moving Average
                    def plot_moving_averages(df):
                        is_dark = is_dark_mode()

                        fig = go.Figure()
                        
                        # Giá đóng cửa gốc
                        fig.add_trace(go.Scatter(
                            x=df['Date'], 
                            y=df['Close'],
                            mode='lines',
                            name='Adj Close',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Tính các MA khác nhau
                        df['MA_2'] = df['Close'].rolling(window=2).mean()
                        df['MA_3'] = df['Close'].rolling(window=3).mean()
                        df['MA_6'] = df['Close'].rolling(window=6).mean()
                        
                        # Thêm các MA vào biểu đồ
                        fig.add_trace(go.Scatter(
                            x=df['Date'], 
                            y=df['MA_2'],
                            mode='lines',
                            name='Naive_MA (2-day)',
                            line=dict(color='orange', width=1.5, dash='dash')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df['Date'], 
                            y=df['MA_3'],
                            mode='lines',
                            name='MA_3_Step (3-day)',
                            line=dict(color='green', width=1.5, dash='dot')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df['Date'], 
                            y=df['MA_6'],
                            mode='lines',
                            name='MA_6_Step (6-day)',
                            line=dict(color='red', width=1.5, dash='dashdot')
                        ))
                        
                        fig.update_layout(
                            title="Giá đóng cửa đã điều chỉnh và đường trung bình động của COP",
                            xaxis_title="Năm",
                            yaxis_title="Giá",
                            height=600,
                            template="plotly_white" if not is_dark else "plotly_dark"

                        )
                        
                        return fig
                    
                    # Hiển thị biểu đồ dựa trên kiểu đã chọn
                    st.markdown("## 📈 Biểu đồ dự báo")
                    
                    # Lọc dữ liệu lịch sử hiển thị
                    history_df = df.tail(history_display_days)
                    
                    # Chọn mô hình phù hợp với kiểu biểu đồ đã chọn
                    if viz_style == "Biểu đồ Prophet":
                        if 'Facebook Prophet' in all_forecasts:
                            st.subheader("Biểu đồ dự báo kiểu Prophet")
                            fig = plot_prophet_style(all_forecasts['Facebook Prophet'], df, 'Facebook Prophet')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Không có kết quả từ mô hình Prophet. Vui lòng chọn mô hình Prophet trong cấu hình.")
                    
                    elif viz_style == "Biểu đồ Holt-Winters (Tối ưu)":
                        hw_model = None
                        for model_name in all_forecasts:
                            if 'Holt-Winters' in model_name and ('standard' in model_name.lower() or 'tiêu chuẩn' in model_name.lower() or 'Hệ số tiêu chuẩn' in model_name):
                                hw_model = model_name
                                break
                        
                        if hw_model:
                            st.subheader("Biểu đồ dự báo Holt-Winters (Hệ số tối ưu)")
                            fig = plot_holt_winters_optimized(all_forecasts[hw_model], df, hw_model)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Không có kết quả từ mô hình Holt-Winters tối ưu. Vui lòng chọn mô hình Holt-Winters trong cấu hình.")
                    
                    elif viz_style == "Biểu đồ Holt-Winters (Tiêu chuẩn)":
                        hw_model = None
                        for model_name in all_forecasts:
                            if 'Holt-Winters' in model_name and ('standard' in model_name.lower() or 'tiêu chuẩn' in model_name.lower() or 'Hệ số tiêu chuẩn' in model_name):
                                hw_model = model_name
                                break
                        
                        if hw_model:
                            st.subheader("Biểu đồ dự báo Holt-Winters (Hệ số tiêu chuẩn)")
                            fig = plot_holt_winters_standard(all_forecasts[hw_model], df, hw_model)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Không có kết quả từ mô hình Holt-Winters tiêu chuẩn. Vui lòng chọn mô hình Holt-Winters trong cấu hình.")
                    
                    elif viz_style == "Biểu đồ Holt (Tham số cố định)":
                        holt_model = 'Holt Fixed (α=0.10, β=0.20)' if 'Holt Fixed (α=0.10, β=0.20)' in all_forecasts else None
                        
                        if holt_model:
                            st.subheader("Biểu đồ dự báo Holt (Tham số cố định)")
                            fig = plot_holt_fixed(all_forecasts[holt_model], df, holt_model)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Không có kết quả từ mô hình Holt với tham số cố định. Vui lòng chọn mô hình Holt trong cấu hình.")
                    
                    elif viz_style == "Biểu đồ Holt (Tham số tối ưu)":
                        holt_model = 'Holt (Double ES)' if 'Holt (Double ES)' in all_forecasts else None
                        
                        if holt_model:
                            st.subheader("Biểu đồ dự báo Holt (Tham số tối ưu)")
                            fig = plot_holt_optimized(all_forecasts[holt_model], df, holt_model)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Không có kết quả từ mô hình Holt với tham số tối ưu. Vui lòng chọn mô hình Holt trong cấu hình.")
                    
                    elif viz_style == "Biểu đồ SES (Alpha tối ưu)":
                        ses_model = 'Simple Exponential Smoothing' if 'Simple Exponential Smoothing' in all_forecasts else None
                        
                        if ses_model and all_forecasts[ses_model].get('alpha', 0) > 0.5:  # Giả sử alpha > 0.5 là tối ưu
                            st.subheader("Biểu đồ dự báo SES (Alpha tối ưu)")
                            fig = plot_ses_optimized(all_forecasts[ses_model], df, ses_model)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Không có kết quả từ mô hình SES với alpha tối ưu. Vui lòng chọn mô hình Exponential Smoothing trong cấu hình.")
                    
                    elif viz_style == "Biểu đồ SES (Alpha cố định)":
                        ses_model = 'Simple Exponential Smoothing' if 'Simple Exponential Smoothing' in all_forecasts else None
                        
                        if ses_model:
                            # Tạo một bản sao của kết quả SES và đặt alpha = 0.1 để hiển thị đúng kiểu
                            ses_fixed = all_forecasts[ses_model].copy()
                            ses_fixed['alpha'] = 0.1
                            
                            st.subheader("Biểu đồ dự báo SES (Alpha cố định)")
                            fig = plot_ses_fixed(ses_fixed, df, ses_model)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Không có kết quả từ mô hình SES. Vui lòng chọn mô hình Exponential Smoothing trong cấu hình.")
                    
                    elif viz_style == "Biểu đồ Moving Average":
                        st.subheader("Biểu đồ Moving Average")
                        fig = plot_moving_averages(df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Hiển thị thông tin chi tiết về mô hình
                    st.markdown("## 📊 Thông tin chi tiết về mô hình")
                    
                    for model_name, forecast in all_forecasts.items():
                        with st.expander(f"📈 {model_name}", expanded=False):
                            # Hiển thị thông số mô hình
                            
                            
                            param_cols = st.columns(4)
                            col_idx = 0
                            
                            if 'alpha' in forecast and forecast['alpha'] is not None:
                                with param_cols[col_idx % 4]:
                                    st.metric("Alpha (α)", f"{forecast['alpha']:.4f}")
                                col_idx += 1
                            
                            if 'beta' in forecast and forecast['beta'] is not None:
                                with param_cols[col_idx % 4]:
                                    st.metric("Beta (β)", f"{forecast['beta']:.4f}")
                                col_idx += 1
                            
                            if 'gamma' in forecast and forecast['gamma'] is not None:
                                with param_cols[col_idx % 4]:
                                    st.metric("Gamma (γ)", f"{forecast['gamma']:.4f}")
                                col_idx += 1
                            
                            if 'seasonal_periods' in forecast:
                                with param_cols[col_idx % 4]:
                                    st.metric("Chu kỳ mùa vụ", f"{forecast['seasonal_periods']}")
                                col_idx += 1
                            
                            if 'trend_type' in forecast:
                                with param_cols[col_idx % 4]:
                                    st.metric("Loại xu hướng", f"{forecast['trend_type']}")
                                col_idx += 1
                            
                            if 'seasonal_type' in forecast:
                                with param_cols[col_idx % 4]:
                                    st.metric("Loại mùa vụ", f"{forecast['seasonal_type']}")
                                col_idx += 1
                            
                            # Hiển thị chỉ số đánh giá
                            if 'errors' in forecast:
                                
                                display_forecast_metrics(forecast, model_name)
                            
                            # Hiển thị dự báo
                            st.markdown("### 📈 Kết quả dự báo")
                            
                            # Tạo DataFrame từ kết quả dự báo
                            forecast_df = pd.DataFrame({
                                'Date': forecast['dates'],
                                'Forecast': forecast['values'],
                                'Lower Bound': forecast['lower'] if 'lower' in forecast else None,
                                'Upper Bound': forecast['upper'] if 'upper' in forecast else None
                            })
                            
                            st.dataframe(forecast_df.head(10), use_container_width=True)
                    
                    # Biểu đồ so sánh tất cả các mô hình
                    st.markdown("## 🔍 So sánh tất cả các mô hình")
                    
                    fig_compare = go.Figure()
                    
                    history_dates = pd.to_datetime(history_df['Date'])
                    history_dates_py = [d.to_pydatetime() if isinstance(d, pd.Timestamp) else d for d in history_dates]
                    
                    # Dữ liệu lịch sử
                    fig_compare.add_trace(go.Scatter(
                        x=history_dates_py, 
                        y=history_df['Close'],
                        name='Dữ liệu lịch sử',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Thêm các mô hình
                    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
                    color_idx = 0
                    
                    for model_name, forecast in all_forecasts.items():
                        fig_compare.add_trace(go.Scatter(
                            x=forecast['dates'], 
                            y=forecast['values'],
                            name=model_name,
                            line=dict(color=colors[color_idx % len(colors)], width=1.5, dash='dash')
                        ))
                        color_idx += 1
                    
                    # Đường dọc đánh dấu bắt đầu dự báo
                    last_date = history_dates_py[-1]
                    fig_compare.add_vline(
                        x=last_date, 
                        line_dash="dash",
                        line_color="red"
                        )

# Thêm annotation riêng
                    fig_compare.add_annotation(
                        x=last_date,
                        y=1,                # đỉnh khung vẽ
                        xref="x",
                        yref="paper",       # 0–1 theo chiều cao figure
                        text="Last real data",
                        showarrow=False,
                        yshift=10           # nhích annotation lên một chút
)
                        
                    
                    fig_compare.update_layout(
                        title=f"So sánh các mô hình dự báo cho {symbol}",
                        xaxis_title="Ngày",
                        yaxis_title="Giá",
                        height=600,
                        template="plotly_white"

                    )
                    
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # Bảng so sánh các mô hình
                    st.markdown("## 📊 Bảng so sánh các mô hình")
                    
                    comparison_data = []
                    
                    for model_name, forecast in all_forecasts.items():
                        row = {
                            'Mô hình': model_name,
                            'Giá cuối dự báo': f"${forecast['values'][-1]:.2f}",
                            'Thay đổi (%)': f"{((forecast['values'][-1] - current_price) / current_price * 100):.2f}%"
                        }
                        
                        if 'errors' in forecast:
                            row['RMSE'] = f"{forecast['errors']['RMSE']:.4f}"
                            row['MAPE'] = f"{forecast['errors']['MAPE']:.2f}%"
                        
                        comparison_data.append(row)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                
                else:
                    st.error("❌ Không có mô hình nào tạo được dự báo. Vui lòng kiểm tra lại cấu hình.")
            
            else:
                st.error("❌ Không đủ dữ liệu để dự báo. Cần ít nhất 30 ngày dữ liệu.")

# ==================== TRANG CHỈ SỐ KỸ THUẬT ====================
elif menu == "📊 Chỉ số kỹ thuật nâng cao":
    st.markdown('<div class="main-header" style="color: white;">📊 Phân tích chỉ số kỹ thuật nâng cao</div>', unsafe_allow_html=True)
    st.info("💡 **Tính năng:** Phân tích đa chỉ báo kỹ thuật: RSI, MACD, Bollinger Bands, MA, EMA, Stochastic, Volume, ATR, ADX, CCI, Williams %R")
    
    col1, col2 = st.columns(2)
    with col1:
        adv_symbol = st.text_input("Mã chứng khoán", value="COP", help="VD: COP, AAPL, SLB")
    
    with col2:
        display_period = st.selectbox(
            "Chọn khung thời gian", 
            ["1 tháng", "3 tháng", "6 tháng", "1 năm", "2 năm", "5 năm"],
            index=3
        )
        
        period_map = {
            "1 tháng": "1mo", "3 tháng": "3mo", "6 tháng": "6mo",
            "1 năm": "1y", "2 năm": "2y", "5 năm": "5y"
        }
        selected_code = period_map[display_period]
    
    st.markdown("### 🔧 Lựa chọn chỉ báo kỹ thuật")
    
    # Chia thành 2 nhóm chỉ báo
    st.markdown("#### 📈 Nhóm 1: Chỉ báo cơ bản")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1: 
        use_rsi = st.checkbox("RSI", value=True, help="Relative Strength Index")
        use_macd = st.checkbox("MACD", value=True, help="Moving Average Convergence Divergence")
        
    with c2: 
        use_bb = st.checkbox("Bollinger Bands", value=True, help="Dải Bollinger")
        use_stoch = st.checkbox("Stochastic", value=True, help="Dao động ngẫu nhiên")
        
    with c3: 
        use_sma = st.checkbox("SMA", value=True, help="Simple Moving Average")
        use_ema = st.checkbox("EMA", value=True, help="Exponential Moving Average")
    
    with c4:
        use_volume = st.checkbox("Volume", value=True, help="Khối lượng giao dịch")
        use_obv = st.checkbox("OBV", value=False, help="On Balance Volume")
    
    st.markdown("#### 📊 Nhóm 2: Chỉ báo nâng cao")
    c5, c6, c7, c8 = st.columns(4)
    
    with c5:
        use_atr = st.checkbox("ATR", value=False, help="Average True Range")
        use_adx = st.checkbox("ADX", value=False, help="Average Directional Index")
    
    with c6:
        use_cci = st.checkbox("CCI", value=False, help="Commodity Channel Index")
        use_williams = st.checkbox("Williams %R", value=False, help="Williams Percent Range")
    
    with c7:
        use_vpt = st.checkbox("VPT", value=False, help="Volume Price Trend")
        use_support_resistance = st.checkbox("Support/Resistance", value=False, help="Hỗ trợ/Kháng cự")
    
    with c8:
        show_candlestick = st.checkbox("Nến Nhật", value=True, help="Biểu đồ nến")
    
    # Tùy chỉnh nâng cao
    with st.expander("⚙️ Tùy chỉnh tham số", expanded=False):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            rsi_period = st.slider("Chu kỳ RSI", min_value=5, max_value=30, value=14, step=1)
            macd_fast = st.slider("MACD Fast", min_value=5, max_value=20, value=12, step=1)
            macd_slow = st.slider("MACD Slow", min_value=20, max_value=40, value=26, step=1)
        
        with adv_col2:
            bb_period = st.slider("Chu kỳ BB", min_value=10, max_value=50, value=20, step=5)
            bb_std = st.slider("BB Std Dev", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
            stoch_period = st.slider("Chu kỳ Stochastic", min_value=5, max_value=30, value=14, step=1)
        
        with adv_col3:
            sma_period = st.slider("Chu kỳ SMA", min_value=5, max_value=200, value=20, step=5)
            ema_period = st.slider("Chu kỳ EMA", min_value=5, max_value=200, value=20, step=5)
            atr_period = st.slider("Chu kỳ ATR", min_value=5, max_value=30, value=14, step=1)
    
    if st.button("🚀 Phân tích kỹ thuật", type="primary"):
        with st.spinner("Đang xử lý dữ liệu và tính toán chỉ số..."):
            try:
                ticker = yf.Ticker(adv_symbol)
                
                short_term_periods = ["1mo", "3mo", "6mo"]
                if selected_code in short_term_periods:
                    download_period = "1y" 
                else:
                    download_period = selected_code 
                
                df = ticker.history(period=download_period)
                
                if df.empty:
                    st.error("❌ Không có dữ liệu cho mã chứng khoán này.")
                else:
                    df = df.reset_index()
                    df = clean_data(df)
                    
                    # Áp dụng TechnicalAnalyzer
                    analyzer = TechnicalAnalyzer(df)
                    df_view = analyzer.df
                    
                    # Lọc dữ liệu hiển thị
                    if selected_code == "1mo": 
                        df_view = df_view.tail(22)
                    elif selected_code == "3mo": 
                        df_view = df_view.tail(65)
                    elif selected_code == "6mo": 
                        df_view = df_view.tail(130)
                    
                    st.success(f"✅ Đã phân tích chỉ số kỹ thuật cho **{adv_symbol.upper()}**")
                    
                    # THỐNG KÊ TỔNG QUAN
                    st.markdown("### 📈 Thống kê tổng quan")
                    
                    current_price = df_view['Close'].iloc[-1]
                    prev_price = df_view['Close'].iloc[-2] if len(df_view) > 1 else current_price
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0
                    
                    col_overview1, col_overview2, col_overview3, col_overview4, col_overview5 = st.columns(5)
                    
                    with col_overview1:
                        st.metric("Giá hiện tại", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
                    
                    with col_overview2:
                        if 'RSI' in df_view.columns:
                            current_rsi = df_view['RSI'].iloc[-1]
                            st.metric("RSI", f"{current_rsi:.2f}")
                    
                    with col_overview3:
                        if 'MACD' in df_view.columns:
                            current_macd = df_view['MACD'].iloc[-1]
                            st.metric("MACD", f"{current_macd:.4f}")
                    
                    with col_overview4:
                        if 'Stoch_%K' in df_view.columns:
                            current_stoch = df_view['Stoch_%K'].iloc[-1]
                            st.metric("Stochastic %K", f"{current_stoch:.2f}")
                    
                    with col_overview5:
                        if 'ATR' in df_view.columns:
                            current_atr = df_view['ATR'].iloc[-1]
                            st.metric("ATR", f"{current_atr:.2f}")
                    
                    # TÓM TẮT TÍN HIỆU
                    st.markdown("### 🎯 Tóm tắt tín hiệu giao dịch")
                    
                    signals = analyzer.get_signals_summary()
                    
                    col_sig1, col_sig2, col_sig3 = st.columns(3)
                    
                    with col_sig1:
                        st.markdown("#### 📊 Chỉ báo động lượng")
                        
                        if 'RSI' in signals:
                            rsi_data = signals['RSI']
                            rsi_color = "red" if rsi_data['signal'] == 'Overbought' else "green" if rsi_data['signal'] == 'Oversold' else "orange"
                            st.markdown(f"**RSI:** {rsi_data['value']:.2f}")
                            st.markdown(f"<span style='color:{rsi_color}'>• {rsi_data['interpretation']}</span>", unsafe_allow_html=True)
                        
                        if 'Stochastic' in signals:
                            stoch_data = signals['Stochastic']
                            stoch_color = "red" if stoch_data['signal'] == 'Overbought' else "green" if stoch_data['signal'] == 'Oversold' else "orange"
                            st.markdown(f"**Stochastic:** {stoch_data['value']:.2f}")
                            st.markdown(f"<span style='color:{stoch_color}'>• {stoch_data['interpretation']}</span>", unsafe_allow_html=True)
                        
                        if 'Williams_%R' in signals:
                            wr_data = signals['Williams_%R']
                            st.markdown(f"**Williams %R:** {wr_data['value']:.2f}")
                            st.markdown(f"• {wr_data['interpretation']}")
                    
                    with col_sig2:
                        st.markdown("#### 📈 Chỉ báo xu hướng")
                        
                        if 'MACD' in signals:
                            macd_data = signals['MACD']
                            macd_color = "green" if macd_data['signal'] == 'Bullish' else "red" if macd_data['signal'] == 'Bearish' else "orange"
                            st.markdown(f"**MACD:** {macd_data['value']:.4f}")
                            st.markdown(f"<span style='color:{macd_color}'>• {macd_data['interpretation']}</span>", unsafe_allow_html=True)
                        
                        if 'ADX' in signals:
                            adx_data = signals['ADX']
                            st.markdown(f"**ADX:** {adx_data['value']:.2f}")
                            st.markdown(f"• {adx_data['interpretation']}")
                        
                        if 'Bollinger_Bands' in signals:
                            bb_data = signals['Bollinger_Bands']
                            st.markdown(f"**BB Position:** {bb_data['position']:.2f}")
                            st.markdown(f"• {bb_data['interpretation']}")
                    
                    with col_sig3:
                        st.markdown("#### 🔄 Chỉ báo khác")
                        
                        if 'CCI' in signals:
                            cci_data = signals['CCI']
                            st.markdown(f"**CCI:** {cci_data['value']:.2f}")
                            st.markdown(f"• {cci_data['interpretation']}")
                        
                        # Tính trung bình khối lượng
                        if 'Volume' in df_view.columns:
                            avg_volume = df_view['Volume'].tail(20).mean()
                            current_volume = df_view['Volume'].iloc[-1]
                            volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1
                            volume_signal = "Cao" if volume_ratio > 1.2 else "Thấp" if volume_ratio < 0.8 else "Bình thường"
                            volume_color = "green" if volume_ratio > 1.2 else "red" if volume_ratio < 0.8 else "orange"
                            st.markdown(f"**Volume:** {current_volume:,.0f}")
                            st.markdown(f"<span style='color:{volume_color}'>• {volume_signal} ({volume_ratio:.2f}x TB)</span>", unsafe_allow_html=True)
                    
                    # BIỂU ĐỒ CHỈ SỐ KỸ THUẬT
                    st.markdown("---")
                    st.markdown("### 📊 Biểu đồ chỉ số kỹ thuật")
                    
                    # Đếm số lượng subplot cần thiết
                    num_subplots = 1  # Giá luôn có
                    if use_volume or use_obv:
                        num_subplots += 1
                    if use_rsi:
                        num_subplots += 1
                    if use_macd:
                        num_subplots += 1
                    if use_stoch:
                        num_subplots += 1
                    if use_atr:
                        num_subplots += 1
                    if use_adx:
                        num_subplots += 1
                    if use_cci:
                        num_subplots += 1
                    if use_williams:
                        num_subplots += 1
                    
                    # Tính row_heights
                    row_heights = [0.4] + [0.6 / (num_subplots - 1)] * (num_subplots - 1) if num_subplots > 1 else [1.0]
                    
                    # Tạo subplot
                    fig = make_subplots(
                        rows=num_subplots, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_heights=row_heights,
                        subplot_titles=[""] * num_subplots
                    )
                    
                    current_row = 1
                    
                    # BIỂU ĐỒ GIÁ
                    if show_candlestick:
                        fig.add_trace(
                            go.Candlestick(
                            x=df_view['Date'],
                            open=df_view['Open'], 
                            high=df_view['High'],
                            low=df_view['Low'], 
                            close=df_view['Close'],
                            name='Giá',
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350'
                        ), row=current_row, col=1)
                    else:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['Close'],
                            name='Giá đóng cửa',
                            line=dict(color='blue', width=2)
                        ), row=current_row, col=1)
                    
                    # ĐƯỜNG TRUNG BÌNH
                    if use_sma and 'SMA_20' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['SMA_20'],
                            name='SMA 20',
                            line=dict(color='orange', width=1.5, dash='dash')
                        ), row=current_row, col=1)
                    
                    if use_ema and 'EMA_12' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['EMA_12'],
                            name='EMA 12',
                            line=dict(color='purple', width=1.5, dash='dot')
                        ), row=current_row, col=1)
                    
                    # BOLLINGER BANDS
                    if use_bb and 'BB_Upper' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['BB_Upper'],
                            name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash'),
                            showlegend=False
                        ), row=current_row, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['BB_Middle'],
                            name='BB Middle',
                            line=dict(color='gray', width=1),
                            showlegend=False
                        ), row=current_row, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['BB_Lower'],
                            name='Bollinger Bands',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(128, 128, 128, 0.1)'
                        ), row=current_row, col=1)
                    
                    # Support/Resistance
                    if use_support_resistance and 'Support' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['Support'],
                            name='Support',
                            line=dict(color='green', width=1, dash='dot'),
                            opacity=0.5
                        ), row=current_row, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['Resistance'],
                            name='Resistance',
                            line=dict(color='red', width=1, dash='dot'),
                            opacity=0.5
                        ), row=current_row, col=1)
                    
                    fig.update_yaxes(title_text="Giá (USD)", row=current_row, col=1)
                    current_row += 1
                    
                    # VOLUME
                    if use_volume or use_obv:
                        if use_volume and 'Volume' in df_view.columns:
                            colors_volume = ['#ef5350' if row['Close'] < row['Open'] else '#26a69a' 
                                           for _, row in df_view.iterrows()]
                            
                            fig.add_trace(go.Bar(
                                x=df_view['Date'], 
                                y=df_view['Volume'],
                                name='Volume',
                                marker_color=colors_volume,
                                opacity=0.7,
                                showlegend=True
                            ), row=current_row, col=1)
                        
                        if use_obv and 'OBV' in df_view.columns:
                            fig.add_trace(go.Scatter(
                                x=df_view['Date'], 
                                y=df_view['OBV'],
                                name='OBV',
                                line=dict(color='purple', width=2),
                                yaxis='y2'
                            ), row=current_row, col=1)
                        
                        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
                        current_row += 1
                    
                    # RSI
                    if use_rsi and 'RSI' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['RSI'],
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ), row=current_row, col=1)
                        
                        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                     annotation_text="Quá mua", annotation_position="right",
                                     row=current_row, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green",
                                     annotation_text="Quá bán", annotation_position="right",
                                     row=current_row, col=1)
                        fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                                     row=current_row, col=1)
                        
                        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
                        current_row += 1
                    
                    # MACD
                    if use_macd and 'MACD' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['MACD'],
                            name='MACD',
                            line=dict(color='blue', width=2)
                        ), row=current_row, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['Signal_Line'],
                            name='Signal Line',
                            line=dict(color='orange', width=1.5)
                        ), row=current_row, col=1)
                        
                        colors_macd = ['#ef5350' if val < 0 else '#26a69a' 
                                     for val in df_view['MACD_Histogram']]
                        fig.add_trace(go.Bar(
                            x=df_view['Date'], 
                            y=df_view['MACD_Histogram'],
                            name='MACD Histogram',
                            marker_color=colors_macd,
                            opacity=0.5
                        ), row=current_row, col=1)
                        
                        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                        current_row += 1
                    
                    # STOCHASTIC
                    if use_stoch and 'Stoch_%K' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['Stoch_%K'],
                            name='Stochastic %K',
                            line=dict(color='deepskyblue', width=2)
                        ), row=current_row, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['Stoch_%D'],
                            name='Stochastic %D',
                            line=dict(color='orange', width=1.5, dash='dash')
                        ), row=current_row, col=1)
                        
                        fig.add_hline(y=80, line_dash="dash", line_color="red",
                                     annotation_text="Quá mua", annotation_position="right",
                                     row=current_row, col=1)
                        fig.add_hline(y=20, line_dash="dash", line_color="green",
                                     annotation_text="Quá bán", annotation_position="right",
                                     row=current_row, col=1)
                        
                        fig.update_yaxes(title_text="Stochastic", range=[0, 100], row=current_row, col=1)
                        current_row += 1
                    
                    # ATR
                    if use_atr and 'ATR' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['ATR'],
                            name='ATR',
                            line=dict(color='brown', width=2)
                        ), row=current_row, col=1)
                        
                        fig.update_yaxes(title_text="ATR", row=current_row, col=1)
                        current_row += 1
                    
                    # ADX
                    if use_adx and 'ADX' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['ADX'],
                            name='ADX',
                            line=dict(color='black', width=2)
                        ), row=current_row, col=1)
                        
                        if '+DI' in df_view.columns:
                            fig.add_trace(go.Scatter(
                                x=df_view['Date'], 
                                y=df_view['+DI'],
                                name='+DI',
                                line=dict(color='green', width=1.5)
                            ), row=current_row, col=1)
                        
                        if '-DI' in df_view.columns:
                            fig.add_trace(go.Scatter(
                                x=df_view['Date'], 
                                y=df_view['-DI'],
                                name='-DI',
                                line=dict(color='red', width=1.5)
                            ), row=current_row, col=1)
                        
                        fig.add_hline(y=25, line_dash="dash", line_color="gray",
                                     annotation_text="Xu hướng mạnh", annotation_position="right",
                                     row=current_row, col=1)
                        
                        fig.update_yaxes(title_text="ADX", row=current_row, col=1)
                        current_row += 1
                    
                    # CCI
                    if use_cci and 'CCI' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['CCI'],
                            name='CCI',
                            line=dict(color='teal', width=2)
                        ), row=current_row, col=1)
                        
                        fig.add_hline(y=100, line_dash="dash", line_color="red",
                                     annotation_text="Quá mua", annotation_position="right",
                                     row=current_row, col=1)
                        fig.add_hline(y=-100, line_dash="dash", line_color="green",
                                     annotation_text="Quá bán", annotation_position="right",
                                     row=current_row, col=1)
                        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                                     row=current_row, col=1)
                        
                        fig.update_yaxes(title_text="CCI", row=current_row, col=1)
                        current_row += 1
                    
                    # WILLIAMS %R
                    if use_williams and 'Williams_%R' in df_view.columns:
                        fig.add_trace(go.Scatter(
                            x=df_view['Date'], 
                            y=df_view['Williams_%R'],
                            name='Williams %R',
                            line=dict(color='darkviolet', width=2)
                        ), row=current_row, col=1)
                        
                        fig.add_hline(y=-20, line_dash="dash", line_color="red",
                                     annotation_text="Quá mua", annotation_position="right",
                                     row=current_row, col=1)
                        fig.add_hline(y=-80, line_dash="dash", line_color="green",
                                     annotation_text="Quá bán", annotation_position="right",
                                     row=current_row, col=1)
                        fig.add_hline(y=-50, line_dash="dot", line_color="gray",
                                     row=current_row, col=1)
                        
                        fig.update_yaxes(title_text="Williams %R", range=[-100, 0], row=current_row, col=1)
                        current_row += 1
                    
                    # CẬP NHẬT LAYOUT
                    fig.update_layout(
                        title=f"Phân tích chỉ số kỹ thuật: {adv_symbol.upper()} ({display_period})",
                        height=200 * num_subplots,
                        xaxis_rangeslider_visible=False,
                        hovermode="x unified",
                        template="plotly_white",
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=1.02, 
                            xanchor="right", 
                            x=1
                        ),
                        margin=dict(t=100, b=50)
                    )
                    
                    fig.update_xaxes(
                        title_text="Thời gian",
                        row=num_subplots, 
                        col=1,
                        rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # BẢNG DỮ LIỆU CHI TIẾT
                    with st.expander("📥 Xem và xuất dữ liệu chi tiết"):
                        # Chọn các cột để hiển thị
                        display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        
                        if use_rsi and 'RSI' in df_view.columns:
                            display_cols.append('RSI')
                        if use_macd and 'MACD' in df_view.columns:
                            display_cols.extend(['MACD', 'Signal_Line'])
                        if use_stoch and 'Stoch_%K' in df_view.columns:
                            display_cols.extend(['Stoch_%K', 'Stoch_%D'])
                        if use_bb and 'BB_Upper' in df_view.columns:
                            display_cols.extend(['BB_Upper', 'BB_Middle', 'BB_Lower'])
                        if use_atr and 'ATR' in df_view.columns:
                            display_cols.append('ATR')
                        if use_adx and 'ADX' in df_view.columns:
                            display_cols.extend(['ADX', '+DI', '-DI'])
                        if use_cci and 'CCI' in df_view.columns:
                            display_cols.append('CCI')
                        if use_williams and 'Williams_%R' in df_view.columns:
                            display_cols.append('Williams_%R')
                        if use_obv and 'OBV' in df_view.columns:
                            display_cols.append('OBV')
                        
                        # Lọc các cột tồn tại
                        display_cols = [col for col in display_cols if col in df_view.columns]
                        
                        st.dataframe(df_view[display_cols].tail(50), use_container_width=True)
                        
                        csv = df_view[display_cols].to_csv(index=False)
                        st.download_button(
                            label="📥 Tải dữ liệu CSV",
                            data=csv,
                            file_name=f"{adv_symbol}_technical_indicators_{display_period}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Lỗi khi phân tích: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# FOOTER
st.markdown("---")
st.markdown("📊 **Phân Tích Thống Kê dự báo cổ phiếu của 3 cô nàng thư giãn**")

# Thêm thông tin về các thư viện trong sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### '' Nắm bắt nhịp đập thị trường bằng góc nhìn vượt thời gian - Nơi ba con người tạo nên tương lai đầu tư '' ")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    **Lưu ý khi dự báo:**
    - Dự báo ngắn hạn (< 30 ngày) thường chính xác hơn
    - Kết hợp nhiều mô hình để có cái nhìn tổng quan
    - Chú ý các chỉ số MAPE, MAE, RMSE
    - MAPE < 10%: Dự báo rất tốt
    - MAPE 10-20%: Dự báo tốt
    - MAPE > 50%: Dự báo kém
    """)

