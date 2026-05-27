import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
import math
import os
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Retail Demand Forecasting | Enterprise Analytics", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Executive KPI Card Styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        text-align: left;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        border-top: 3px solid #2563EB;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-subtext {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }
    
    .text-green { color: #10B981; }
    .text-red { color: #EF4444; }
    .text-gray { color: #94A3B8; }
    .text-blue { color: #3B82F6; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    hr {
        border-color: #E2E8F0;
        margin: 1.5rem 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        margin-right: 10px;
    }
    .status-badge-active {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    
    /* Clean up tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 500;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] {
        color: #0F172A;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA CACHING ---
@st.cache_data
def load_data():
    data_path = "data/train.csv"
    if not os.path.exists(data_path):
        return None, data_path
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month_name'] = df['date'].dt.month_name()
    return df, data_path

@st.cache_resource
def train_forecast_model(df_f, lookback=30):
    data = df_f['sales'].values.astype(float)
    X, Y = [], []
    for i in range(lookback, len(data)):
        window = data[i-lookback:i].tolist()
        dow = df_f.index[i].dayofweek
        month = df_f.index[i].month
        X.append(window + [dow, month])
        Y.append(data[i])
        
    X, Y = np.array(X), np.array(Y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = Y[:train_size], Y[train_size:]
    
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = math.sqrt(mean_squared_error(y_test, test_pred))
    mape = mean_absolute_percentage_error(y_test, test_pred) * 100
    
    return model, y_test, test_pred, mae, rmse, mape, data

df, data_path = load_data()
if df is None:
    st.markdown("<h2 style='color:#0F172A;font-weight:700;'>Setup Required</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;'>The dataset is not present. Follow the steps below to get started.</p>", unsafe_allow_html=True)
    st.markdown("---")
    col_setup1, col_setup2 = st.columns([1, 1])
    with col_setup1:
        st.markdown("""
        **Step 1 — Download the dataset**

        Download `train.csv` from the Kaggle competition:
        [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)
        """)
        st.markdown("""
        **Step 2 — Place the file**

        Move the downloaded file to this path in the project:
        ```
        data/train.csv
        ```
        """)
        st.markdown("""
        **Step 3 — Reload the app**

        Refresh this page. The dashboard will load automatically.
        """)
    with col_setup2:
        st.info(f"Expected path: `{os.path.abspath(data_path)}`")
    st.stop()


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Analytics Controls")
    store_id = st.selectbox("Select Store", sorted(df['store'].unique()), index=0)
    item_id = st.selectbox("Select Product", sorted(df['item'].unique()), index=0)
    forecast_horizon = st.slider("Forecast Horizon (Days)", 7, 30, 14)
    
    st.markdown("---")
    st.markdown("### Scenario Planning")
    promo_uplift = st.slider("Demand Uplift Simulation (%)", min_value=-20, max_value=50, value=0, step=5)
    
    st.markdown("---")
    st.markdown("<div style='font-size: 0.75rem; color: #94A3B8;'>Enterprise Forecasting System<br>Version 3.1.0 (Production)</div>", unsafe_allow_html=True)

# --- HEADER SECTION ---
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"<div class='status-badge status-badge-active'>Model Deployed - Online</div> <div class='status-badge'>Last Updated: {current_time}</div>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #0F172A; font-weight: 700; margin-bottom: 0;'>Retail Demand Forecasting</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 1rem;'>AI-driven predictive analytics and scenario simulation platform.</p>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

# --- DATA PREP ---
df_f = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()
df_f = df_f.sort_values('date')
df_f.set_index('date', inplace=True)

# Model Training
with st.spinner("Processing analytics..."):
    lookback = 30
    model, y_test_actual, test_pred, mae, rmse, mape, raw_sales_data = train_forecast_model(df_f, lookback)

# Future Prediction Logic
future_preds_base = []
future_preds_scenario = []
rolling_window = raw_sales_data[-lookback:].copy()
rolling_list = rolling_window.tolist()
last_date = df_f.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

for future_date in future_dates:
    dow = future_date.dayofweek
    month = future_date.month
    features = rolling_list[-lookback:] + [dow, month]
    pred = model.predict([features])[0]
    
    future_preds_base.append(pred)
    future_preds_scenario.append(pred * (1 + promo_uplift/100))
    rolling_list.append(pred)

# --- KPIs (BUSINESS METRICS ONLY) ---
current_month_sales = int(df_f['sales'][-30:].sum())
prev_month_sales = int(df_f['sales'][-60:-30].sum())
mom_growth = ((current_month_sales - prev_month_sales) / prev_month_sales) * 100 if prev_month_sales else 0
avg_daily_sales = int(df_f['sales'][-30:].mean())
forecasted_volume = int(sum(future_preds_scenario))

def get_trend_html(val, is_percentage=True, invert_color=False):
    if val > 0:
        color = "text-red" if invert_color else "text-green"
        arrow = "&uarr;"
    elif val < 0:
        color = "text-green" if invert_color else "text-red"
        arrow = "&darr;"
    else:
        color = "text-gray"
        arrow = "&minus;"
    fmt = f"{abs(val):.1f}%" if is_percentage else f"{abs(val):.2f}"
    return f"<span class='{color}'>{arrow} {fmt}</span>"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Monthly Sales Volume</div>
        <div class="metric-value">{current_month_sales:,}</div>
        <div class="metric-subtext">{get_trend_html(mom_growth)} vs prior period</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Daily Sales</div>
        <div class="metric-value">{avg_daily_sales:,}</div>
        <div class="metric-subtext"><span class='text-gray'>Units / day (30d avg)</span></div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Forecasted Demand ({forecast_horizon}d)</div>
        <div class="metric-value">{forecasted_volume:,}</div>
        <div class="metric-subtext"><span class='text-blue'>Scenario Projected</span></div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Demand Growth Rate</div>
        <div class="metric-value">{mom_growth:+.1f}%</div>
        <div class="metric-subtext">{get_trend_html(mom_growth)} MoM Trend</div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN HERO ANALYTICS SECTION ---
st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #0F172A; font-weight: 600; margin-bottom: 0px;'>Demand Forecasting Overview</h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 0.85rem; color: #64748B;'>Recent historical actuals against machine learning projected baseline and scenario adjustments.</p>", unsafe_allow_html=True)

fig_hero = go.Figure()

context_dates = df_f.index[-60:]
context_sales = df_f['sales'][-60:]

fig_hero.add_trace(go.Scatter(x=context_dates, y=context_sales, mode='lines', 
                              name='Historical Actuals', line=dict(color='#64748B', width=2)))

fig_hero.add_trace(go.Scatter(x=future_dates, y=future_preds_base, mode='lines', 
                              name='Baseline Forecast', line=dict(color='#2563EB', width=3)))
                              
if promo_uplift != 0:
    fig_hero.add_trace(go.Scatter(x=future_dates, y=future_preds_scenario, mode='lines', 
                                  name='Scenario Forecast', line=dict(color='#10B981', width=3, dash='dot')))

# Approximate Confidence Interval Overlap
upper_bound = [p + (p * (mape/100)) for p in future_preds_base]
lower_bound = [p - (p * (mape/100)) for p in future_preds_base]

fig_hero.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates)[::-1],
    y=upper_bound + lower_bound[::-1],
    fill='toself',
    fillcolor='rgba(37, 99, 235, 0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    name='Confidence Interval'
))

fig_hero.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    hovermode="x unified",
    xaxis_title="", yaxis_title="Units",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=10, b=0),
    height=380
)
fig_hero.update_xaxes(showgrid=False, linecolor='#CBD5E1')
fig_hero.update_yaxes(showgrid=True, gridcolor='#E2E8F0', linecolor='#CBD5E1')
st.plotly_chart(fig_hero, use_container_width=True, key="hero_chart")

st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# --- ANALYTICS TAB SYSTEM ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Historical Analytics", 
    "Forecasting Analytics", 
    "Model Evaluation", 
    "Scenario Planning"
])

PLOT_BG = 'rgba(0,0,0,0)'
GRID_COLOR = '#E2E8F0'

with tab1:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #0F172A; margin-top: 10px;'>Historical Sales & Seasonality</div>", unsafe_allow_html=True)
    
    fig_hist = px.line(df_f.reset_index(), x='date', y='sales', color_discrete_sequence=['#94A3B8'])
    
    df_f['rolling_avg'] = df_f['sales'].rolling(window=30).mean()
    fig_hist.add_trace(go.Scatter(x=df_f.index, y=df_f['rolling_avg'], mode='lines', 
                             name='30-Day Moving Avg', line=dict(color='#2563EB', width=2)))
                             
    fig_hist.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        xaxis_title="", yaxis_title="Units Sold",
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320
    )
    fig_hist.update_xaxes(showgrid=False, linecolor='#CBD5E1')
    fig_hist.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, linecolor='#CBD5E1')
    st.plotly_chart(fig_hist, use_container_width=True, key="hist_chart")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-top: 10px; text-transform: uppercase;'>Weekly Demand Distribution</div>", unsafe_allow_html=True)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig_day = px.box(df_f, x='day_of_week', y='sales', category_orders={"day_of_week": day_order},
                         color_discrete_sequence=['#64748B'])
        fig_day.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, xaxis_title="", yaxis_title="Sales Volume", margin=dict(l=0, r=0, t=10, b=0), height=300)
        fig_day.update_xaxes(showgrid=False, linecolor='#CBD5E1')
        fig_day.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, linecolor='#CBD5E1')
        st.plotly_chart(fig_day, use_container_width=True, key="day_chart")
        
    with col_b:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-top: 10px; text-transform: uppercase;'>Monthly Demand Distribution</div>", unsafe_allow_html=True)
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        fig_month = px.box(df_f, x='month_name', y='sales', category_orders={"month_name": month_order},
                           color_discrete_sequence=['#3B82F6'])
        fig_month.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, xaxis_title="", yaxis_title="", margin=dict(l=0, r=0, t=10, b=0), height=300)
        fig_month.update_xaxes(showgrid=False, linecolor='#CBD5E1')
        fig_month.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, linecolor='#CBD5E1')
        st.plotly_chart(fig_month, use_container_width=True, key="month_chart")

with tab2:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #0F172A; margin-top: 10px;'>Demand Trajectory & Confidence</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: #64748B;'>Analyzing prediction boundaries and forecast variance over the selected horizon.</p>", unsafe_allow_html=True)

    # Rebuild as separate figure to avoid DuplicateElementId
    fig_forecast_tab = go.Figure()
    fig_forecast_tab.add_trace(go.Scatter(x=context_dates, y=context_sales, mode='lines',
                                name='Historical Actuals', line=dict(color='#64748B', width=2)))
    fig_forecast_tab.add_trace(go.Scatter(x=future_dates, y=future_preds_base, mode='lines',
                                name='Baseline Forecast', line=dict(color='#2563EB', width=3)))
    if promo_uplift != 0:
        fig_forecast_tab.add_trace(go.Scatter(x=future_dates, y=future_preds_scenario, mode='lines',
                                    name='Scenario Forecast', line=dict(color='#10B981', width=3, dash='dot')))
    fig_forecast_tab.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself', fillcolor='rgba(37, 99, 235, 0.1)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', name='Confidence Interval'
    ))
    fig_forecast_tab.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', xaxis_title='', yaxis_title='Units',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=10, b=0), height=420
    )
    fig_forecast_tab.update_xaxes(showgrid=False, linecolor='#CBD5E1')
    fig_forecast_tab.update_yaxes(showgrid=True, gridcolor='#E2E8F0', linecolor='#CBD5E1')
    st.plotly_chart(fig_forecast_tab, use_container_width=True, key="forecast_tab_chart")

with tab3:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #0F172A; margin-top: 10px;'>Technical Model Evaluation</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: #64748B;'>Machine learning performance metrics on holdout test set.</p>", unsafe_allow_html=True)
    
    confidence_score = max(0, 100 - mape)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 2px solid #94A3B8; padding: 12px 16px;">
            <div class="metric-label">Mean Absolute Error</div>
            <div class="metric-value" style="font-size: 1.5rem;">{mae:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 2px solid #94A3B8; padding: 12px 16px;">
            <div class="metric-label">Root Mean Sq Error</div>
            <div class="metric-value" style="font-size: 1.5rem;">{rmse:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 2px solid #94A3B8; padding: 12px 16px;">
            <div class="metric-label">MAPE</div>
            <div class="metric-value" style="font-size: 1.5rem;">{mape:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 2px solid #10B981; padding: 12px 16px;">
            <div class="metric-label">Prediction Confidence</div>
            <div class="metric-value" style="font-size: 1.5rem;">{confidence_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-top: 15px; text-transform: uppercase;'>Actual vs Predicted (Holdout Set)</div>", unsafe_allow_html=True)
        test_dates = df_f.index[-len(y_test_actual):]
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=test_dates, y=y_test_actual, mode='lines', name='Actual', line=dict(color='#94A3B8', width=2)))
        fig_eval.add_trace(go.Scatter(x=test_dates, y=test_pred, mode='lines', name='Predicted', line=dict(color='#3B82F6', width=2, dash='solid')))
        
        fig_eval.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
            hovermode="x unified",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300
        )
        fig_eval.update_xaxes(showgrid=False, linecolor='#CBD5E1')
        fig_eval.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, linecolor='#CBD5E1')
        st.plotly_chart(fig_eval, use_container_width=True, key="eval_chart")
        
    with col_d:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-top: 15px; text-transform: uppercase;'>Residual Error Distribution</div>", unsafe_allow_html=True)
        residuals = y_test_actual - test_pred
        fig_res = px.histogram(x=residuals, nbins=30, color_discrete_sequence=['#64748B'])
        fig_res.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Prediction Error", yaxis_title="Frequency", height=300)
        fig_res.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
        fig_res.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
        st.plotly_chart(fig_res, use_container_width=True, key="residual_chart")

with tab4:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #0F172A; margin-top: 10px;'>Demand Uplift Simulation</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: #64748B;'>Analyze the potential volume and revenue impact of promotional campaigns.</p>", unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.markdown("""
        <div style="background-color: #FFFFFF; padding: 16px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 1rem;">
            <div style="font-size: 0.75rem; font-weight: 600; color: #64748B; text-transform: uppercase;">Simulation Active</div>
            <div style="font-size: 0.85rem; color: #334155; margin-top: 8px; line-height: 1.4;">Baseline demand is modeled at steady state. Use the sidebar controls to simulate targeted marketing uplifts or external demand shocks.</div>
        </div>
        """, unsafe_allow_html=True)
        
        impact = sum(future_preds_scenario) - sum(future_preds_base)
        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid #10B981;">
            <div class="metric-label">Estimated Volume Impact</div>
            <div class="metric-value">{'+' if impact > 0 else ''}{int(impact):,}</div>
            <div class="metric-subtext">Incremental units generated</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_s2:
        df_scenario = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Baseline Forecast': np.round(future_preds_base, 1),
            'Scenario Forecast': np.round(future_preds_scenario, 1),
            'Delta (Units)': np.round(np.array(future_preds_scenario) - np.array(future_preds_base), 1)
        })
        
        st.dataframe(df_scenario, use_container_width=True, hide_index=True)
