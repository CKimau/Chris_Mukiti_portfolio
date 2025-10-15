# streamlit_portfolio.py
# Light Theme Supply Chain Portfolio - Handles Missing Dependencies
# Run: `streamlit run streamlit_portfolio.py`

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import sqlite3

# Try to import scikit-learn, but provide fallback if not available
try:
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ Scikit-learn not available. Some advanced features will be limited.")

# Try to import Prophet, but provide fallback if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Try to import plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ---------------------------
# Page config & Theme
# ---------------------------
st.set_page_config(
    page_title="Chris Kimau — Supply Chain Forecasting & Analytics Specialist",
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light professional color scheme
PRIMARY = "#2563EB"           # Professional blue
ACCENT = "#059669"            # Professional green
SECONDARY = "#7C3AED"         # Purple
BG = "#FFFFFF"                # White background
CARD = "#F8FAFC"              # Light gray cards
TEXT = "#1E293B"              # Dark text for better readability
SUBTEXT = "#475569"           # Medium gray subtext
BORDER = "#E2E8F0"            # Light border

st.markdown(f"""
<style>
:root {{
    --primary: {PRIMARY};
    --accent: {ACCENT};
    --secondary: {SECONDARY};
    --bg: {BG};
    --card: {CARD};
    --text: {TEXT};
    --subtext: {SUBTEXT};
    --border: {BORDER};
}}

* {{
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    line-height: 1.6;
}}

html, body, [class*="css"] {{
    background: {BG};
    color: {TEXT};
    background-attachment: fixed;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {TEXT} !important;
    font-weight: 600;
    line-height: 1.3;
}}

p, li, div {{
    color: {SUBTEXT} !important;
    font-size: 1.05rem;
    line-height: 1.7;
}}

header[data-testid="stHeader"] {{
    background: {BG};
    border-bottom: 1px solid {BORDER};
}}

.stApp {{
    background: transparent;
}}

.readable-text {{
    color: {SUBTEXT} !important;
    font-size: 1.1rem;
    line-height: 1.7;
    margin-bottom: 1rem;
}}

.neon-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    margin-bottom: 24px;
}}

.neon-card:hover {{
    border-color: {PRIMARY};
    box-shadow: 0 8px 30px rgba(37, 99, 235, 0.12);
    transform: translateY(-2px);
}}

.stButton>button {{
    background: linear-gradient(90deg, {PRIMARY}, #1D4ED8);
    color: white;
    border: none;
    padding: 14px 28px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
}}

.stButton>button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
}}

.badge {{
    background: linear-gradient(90deg, {ACCENT}, #047857);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9em;
}}

.tech-tag {{
    background: rgba(37, 99, 235, 0.08);
    color: {PRIMARY};
    padding: 6px 14px;
    border-radius: 12px;
    border: 1px solid rgba(37, 99, 235, 0.2);
    font-size: 0.9em;
    margin: 4px;
    font-weight: 500;
}}

.achievement-card {{
    background: rgba(5, 150, 105, 0.08);
    border: 1px solid rgba(5, 150, 105, 0.2);
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
}}

.role-highlight {{
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.08), rgba(37, 99, 235, 0.04));
    border-left: 4px solid {PRIMARY};
    padding: 20px;
    border-radius: 12px;
    margin: 20px 0;
}}

.supply-chain-feature {{
    background: {CARD};
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
    border: 1px solid {BORDER};
}}

.dashboard-preview {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
    background: white;
    transition: all 0.3s ease;
}}

.dashboard-preview:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}}

/* Hide Streamlit default elements */
footer {{visibility: hidden;}}
#MainMenu {{visibility: hidden;}}

/* Improve readability for dataframes */
.stDataFrame {{
    border-radius: 8px;
    border: 1px solid {BORDER};
}}

/* Better contrast for metrics */
[data-testid="metric-container"] {{
    background: {CARD};
    border-radius: 12px;
    padding: 16px;
    border: 1px solid {BORDER};
}}

/* Conditional formatting for metrics */
.metric-good {{
    border-left: 4px solid {ACCENT};
}}

.metric-warning {{
    border-left: 4px solid #F59E0B;
}}

.metric-poor {{
    border-left: 4px solid #EF4444;
}}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper functions
# ---------------------------
def get_cv_bytes():
    cv_content = """CHRIS KIMAU
Supply Chain Forecasting & Analytics Specialist

CONTACT
Mobile: +254706109248
Email: kimauchris0@gmail.com
LinkedIn: linkedin.com/in/chrismukitikimau

PROFESSIONAL SUMMARY
Supply Chain and Data Science professional with 6+ years of expertise in demand forecasting, inventory optimization, and logistics planning. Proven track record of implementing AI-driven forecasting solutions and creating interactive dashboards that reduce costs, improve service levels, and optimize supply chain operations.

CORE COMPETENCIES
• Demand Forecasting & Planning
• Inventory Optimization
• Supply Chain Analytics
• Tableau & Power BI Dashboards
• Logistics & Distribution
• Machine Learning & AI
• SAP Systems (MM, PP, SD)
• Data Visualization

TECHNICAL SKILLS
• BI Tools: Tableau, Power BI, Streamlit
• Programming: Python, SQL, R
• Machine Learning: Prophet, ARIMA, Scikit-learn
• Data Visualization: Advanced dashboard development
• Supply Chain Systems: SAP, Inventory Management
• Tools: Git, AWS, Docker, Advanced Excel

EDUCATION & CERTIFICATIONS
• Moringa School: Data Science, Machine Learning & AI
• IBM: Data Science Professional Certificate
• The Catholic University of Eastern Africa: Bachelor of Commerce

PROFESSIONAL EXPERIENCE

Warehouse & Inventory Manager | Skanem Africa (Oct 2024-Present)
• Implement demand forecasting pipelines and inventory optimization dashboards
• Develop Tableau and Power BI reports for supply chain performance monitoring
• Oversee finished goods inventory and SKU-level tracking systems
• Lead process improvements for supply chain efficiency

Supply Chain Analyst | Mabati Rolling Mills (Jan 2024-Oct 2024)
• Developed AI-driven demand forecasting models reducing stockouts by 20%
• Created interactive Power BI dashboards improving sales efficiency by 15%
• Optimized inventory levels, reducing excess stock by 35%
• Enhanced data-driven decision making across supply chain functions

KEY ACHIEVEMENTS
• 20% reduction in stockouts through predictive tracking systems
• 15% revenue growth through data-driven sales strategies
• 35% reduction in excess inventory through optimization algorithms
• 10% logistics cost reduction via optimized contract negotiations
"""
    return cv_content.encode('utf-8')

def create_supply_chain_skill_chart():
    skills = {
        'Demand Forecasting': 92,
        'Tableau/Power BI': 90,
        'Inventory Optimization': 88,
        'Supply Chain Analytics': 87,
        'Logistics Planning': 85,
        'Machine Learning': 86,
        'SAP Systems': 84,
        'Python/SQL': 89
    }
    
    chart_data = pd.DataFrame({
        'Skill': list(skills.keys()),
        'Level': list(skills.values())
    })
    
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Level:Q', title='Proficiency Level', scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('Skill:N', title='', sort='-x'),
        color=alt.Color('Level:Q', scale=alt.Scale(range=[PRIMARY, ACCENT]), legend=None)
    ).properties(height=400, title='Supply Chain & Analytics Skills')
    
    return chart

def safe_mape(y_true, y_pred):
    """Custom MAPE calculation that doesn't require sklearn"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def simple_rmse(y_true, y_pred):
    """Custom RMSE calculation that doesn't require sklearn"""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def simple_r2(y_true, y_pred):
    """Custom R² calculation that doesn't require sklearn"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def create_sample_forecast_data():
    """Generate sample forecast data"""
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    actual = [1200, 1350, 1100, 1450, 1300, 1400, 1250, 1500, 1350, 1420, 1280, 1480]
    forecast = [1150, 1300, 1150, 1400, 1320, 1380, 1280, 1480, 1370, 1450, 1300, 1500]
    
    return pd.DataFrame({
        'Month': dates,
        'Actual': actual,
        'Forecast': forecast
    })

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100

def get_metric_color(metric, value):
    """Determine color based on metric thresholds"""
    if metric == 'mape':
        return ACCENT if value < 10 else '#F59E0B' if value < 20 else '#EF4444'
    elif metric == 'rmse':
        # For RMSE, we need context about the data scale, so we'll use a simple approach
        return ACCENT if value < 100 else '#F59E0B' if value < 200 else '#EF4444'
    elif metric == 'r2':
        return ACCENT if value > 0.7 else '#F59E0B' if value > 0.5 else '#EF4444'
    return PRIMARY

# ---------------------------
# Forecasting App Functions
# ---------------------------
def forecasting_app():
    st.header("📈 Supply Chain Forecasting Demo")
    
    st.markdown(f"""
    <div class='neon-card'>
        <h3 style='color: {PRIMARY}; margin-bottom: 1rem;'>AI-Powered Demand Forecasting</h3>
        <div class='readable-text'>
        This interactive demo showcases advanced supply chain forecasting capabilities using machine learning. 
        Upload your historical data or use our sample data to generate accurate demand forecasts with performance metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("1. Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file with historical data", 
                                   type=["csv", "xlsx"], 
                                   key="forecast_upload")

    # Initialize with empty dataframe
    df = pd.DataFrame()
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = df
            st.success("✅ Data loaded successfully")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()
    else:
        # Use sample data if no file uploaded
        st.info("💡 Using sample supply chain data. Upload your own CSV/Excel file for custom analysis.")
        df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=24, freq='M'),
            'Demand': [1200, 1350, 1100, 1450, 1300, 1400, 1250, 1500, 1350, 1420, 1280, 1480, 
                      1550, 1400, 1600, 1450, 1520, 1380, 1480, 1620, 1450, 1580, 1420, 1650],
            'Product': ['Product_A'] * 12 + ['Product_B'] * 12
        })
        st.dataframe(df.head())

    if not df.empty:
        st.subheader("2. Configure Your Forecast")
        
        # Column selection
        cols = st.columns(2)
        with cols[0]:
            date_col = st.selectbox("Select Date Column", 
                                  df.columns, 
                                  key="forecast_date_col")
        with cols[1]:
            numeric_cols = df.select_dtypes(include='number').columns
            value_col = st.selectbox("Select Value Column", 
                                   numeric_cols if len(numeric_cols) > 0 else df.columns, 
                                   key="forecast_value_col")
        
        # Item filter
        item_col = st.selectbox("Filter by Item Column (optional)", 
                              ["No filter"] + [c for c in df.columns if c not in [date_col, value_col]],
                              key="forecast_item_col")
        
        selected_items = []
        if item_col != "No filter":
            selected_items = st.multiselect("Select items to forecast (select one or more)", 
                                          df[item_col].unique(),
                                          key="forecast_item_select")
            st.info(f"Selected {len(selected_items)} items for forecasting")

        # Prepare data
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            
            if selected_items:
                df = df[df[item_col].isin(selected_items)]
                st.success(f"Filtered data for {len(selected_items)} items")
            else:
                st.info("No specific items selected. Using all available data.")
            
            df = df[[date_col, value_col] + ([item_col] if item_col != "No filter" else [])]
            df = df.rename(columns={date_col: "ds", value_col: "y"})
            df = df.dropna().sort_values("ds")
            
            # Get the latest date from the data
            last_date = df['ds'].max()
            
        except Exception as e:
            st.error(f"❌ Error preparing data: {e}")
            st.stop()

        st.subheader("3. Forecast Settings")
        
        # Forecast configuration
        method = st.radio("Forecasting method", 
                         ["Prophet (recommended)", "Simple Trend"],
                         horizontal=True)
        
        cols = st.columns(2)
        with cols[0]:
            # Calculate default forecast end date (6 months from last date)
            default_end_date = last_date + pd.DateOffset(months=6)
            
            # Create date input for forecast end date
            forecast_end_date = st.date_input(
                "Select Forecast End Date",
                min_value=last_date + pd.Timedelta(days=1),
                max_value=datetime(2028, 12, 31),
                value=default_end_date
            )
            
            # Convert to pandas Timestamp
            forecast_end_date = pd.Timestamp(forecast_end_date)
        with cols[1]:
            freq = st.radio("Frequency", 
                          ["Daily", "Weekly", "Monthly"], 
                          horizontal=True)
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Creating forecast..."):
                try:
                    # Calculate horizon based on selected end date
                    last_date = df['ds'].max()
                    date_range = pd.date_range(start=last_date, end=forecast_end_date, freq='D')
                    
                    if freq == "Daily":
                        horizon = len(date_range) - 1
                    elif freq == "Weekly":
                        horizon = len(date_range) // 7
                    else:  # Monthly
                        horizon = (forecast_end_date.year - last_date.year) * 12 + \
                                 (forecast_end_date.month - last_date.month)
                    
                    # Ensure at least 1 period
                    horizon = max(1, horizon)
                    
                    # Handle multiple items forecasting
                    all_forecasts = []
                    
                    if item_col != "No filter" and selected_items:
                        # Forecast for each selected item individually
                        for item in selected_items:
                            st.write(f"🔮 Forecasting for: {item}")
                            
                            # Filter data for current item
                            item_data = df[df[item_col] == item][['ds', 'y']].copy()
                            
                            if len(item_data) < 2:
                                st.warning(f"Not enough data for {item}. Skipping.")
                                continue
                            
                            # Simple forecasting if Prophet not available
                            if not PROPHET_AVAILABLE or method == "Simple Trend":
                                # Simple linear trend forecast
                                item_data['days'] = (item_data['ds'] - item_data['ds'].min()).dt.days
                                trend = np.polyfit(item_data['days'], item_data['y'], 1)
                                
                                # Create future dates
                                future_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
                                future_days = (future_dates - item_data['ds'].min()).days
                                
                                future_forecast = trend[0] * future_days + trend[1]
                                
                                # Create result dataframe
                                future_df = pd.DataFrame({
                                    'ds': future_dates,
                                    'yhat': future_forecast,
                                    'yhat_lower': future_forecast * 0.9,
                                    'yhat_upper': future_forecast * 1.1
                                })
                                
                                result = pd.merge(item_data, future_df, on='ds', how='outer')
                                result[item_col] = item
                                all_forecasts.append(result)
                                
                            else:
                                # Prophet forecasting
                                m = Prophet()
                                m.fit(item_data)
                                
                                # Map frequency to Prophet frequency codes
                                freq_map = {
                                    "Daily": "D",
                                    "Weekly": "W",
                                    "Monthly": "M"
                                }
                                
                                future = m.make_future_dataframe(periods=horizon, freq=freq_map[freq])
                                forecast = m.predict(future)
                                
                                # Merge actuals and forecast
                                result = pd.merge(item_data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                                                on='ds', how='outer')
                                
                                # Add item information
                                result[item_col] = item
                                all_forecasts.append(result)
                    
                    else:
                        # Single time series forecasting (no item grouping)
                        if not PROPHET_AVAILABLE or method == "Simple Trend":
                            # Simple trend for single series
                            df_simple = df[['ds', 'y']].copy()
                            df_simple['days'] = (df_simple['ds'] - df_simple['ds'].min()).dt.days
                            trend = np.polyfit(df_simple['days'], df_simple['y'], 1)
                            
                            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
                            future_days = (future_dates - df_simple['ds'].min()).days
                            
                            future_forecast = trend[0] * future_days + trend[1]
                            
                            future_df = pd.DataFrame({
                                'ds': future_dates,
                                'yhat': future_forecast,
                                'yhat_lower': future_forecast * 0.9,
                                'yhat_upper': future_forecast * 1.1
                            })
                            
                            result = pd.merge(df_simple[['ds', 'y']], future_df, on='ds', how='outer')
                            if item_col != "No filter":
                                result[item_col] = "All Items"
                            all_forecasts.append(result)
                        else:
                            # Prophet for single series
                            m = Prophet()
                            m.fit(df[['ds', 'y']])
                            
                            freq_map = {
                                "Daily": "D",
                                "Weekly": "W",
                                "Monthly": "M"
                            }
                            
                            future = m.make_future_dataframe(periods=horizon, freq=freq_map[freq])
                            forecast = m.predict(future)
                            
                            result = pd.merge(df[['ds', 'y']], forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                                            on='ds', how='outer')
                            if item_col != "No filter":
                                result[item_col] = "All Items"
                            all_forecasts.append(result)
                    
                    # Combine all forecasts
                    if not all_forecasts:
                        st.error("No forecasts generated. Check your data and selections.")
                        st.stop()
                    
                    combined_result = pd.concat(all_forecasts, ignore_index=True)
                    
                    # Ensure no negative forecasts
                    combined_result['yhat'] = combined_result['yhat'].clip(lower=0)
                    combined_result['yhat_lower'] = combined_result['yhat_lower'].clip(lower=0)
                    combined_result['yhat_upper'] = combined_result['yhat_upper'].clip(lower=0)
                    
                    # Filter to show only up to the selected end date
                    combined_result = combined_result[combined_result['ds'] <= forecast_end_date]
                    
                    # Forecast Preview
                    st.subheader("📋 Forecast Preview")
                    preview_df = combined_result[combined_result['ds'] > last_date].copy()
                    
                    # Display preview with item information
                    display_cols = ['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']
                    if item_col != "No filter":
                        display_cols.insert(0, item_col)
                    
                    st.dataframe(preview_df[display_cols].head(10).style.format({
                        'yhat': '{:.2f}',
                        'yhat_lower': '{:.2f}',
                        'yhat_upper': '{:.2f}',
                        'y': '{:.2f}'
                    }))
                    
                    # Visualization for multiple items
                    st.subheader("📊 Forecast Results")
                    
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure()
                        
                        # Color palette for multiple items
                        colors = px.colors.qualitative.Set1
                        
                        if item_col != "No filter" and selected_items:
                            # Plot each item separately
                            for i, item in enumerate(selected_items):
                                item_data = combined_result[combined_result[item_col] == item]
                                
                                color = colors[i % len(colors)]
                                
                                # Actual values
                                actuals = item_data.dropna(subset=['y'])
                                if not actuals.empty:
                                    fig.add_trace(go.Scatter(
                                        x=actuals['ds'], y=actuals['y'],
                                        name=f'{item} - Actual',
                                        line=dict(color=color),
                                        mode='lines+markers',
                                        opacity=0.7
                                    ))
                                
                                # Forecast
                                forecasts = item_data[item_data['ds'] > last_date]
                                if not forecasts.empty:
                                    fig.add_trace(go.Scatter(
                                        x=forecasts['ds'], y=forecasts['yhat'],
                                        name=f'{item} - Forecast',
                                        line=dict(color=color, dash='dash'),
                                        opacity=0.9
                                    ))
                        else:
                            # Single time series
                            # Actual values
                            actuals = combined_result.dropna(subset=['y'])
                            if not actuals.empty:
                                fig.add_trace(go.Scatter(
                                    x=actuals['ds'], y=actuals['y'],
                                    name='Actual',
                                    line=dict(color=colors[0]),
                                    mode='lines+markers'
                                ))
                            
                            # Forecast
                            forecasts = combined_result[combined_result['ds'] > last_date]
                            if not forecasts.empty:
                                fig.add_trace(go.Scatter(
                                    x=forecasts['ds'], y=forecasts['yhat'],
                                    name='Forecast',
                                    line=dict(color=colors[1])
                                ))
                        
                        # Add vertical line to separate historical data and forecast
                        fig.add_vline(x=last_date.timestamp() * 1000, 
                                    line_dash="dash", 
                                    line_color="green",
                                    annotation_text="Forecast Start",
                                    annotation_position="top left")
                        
                        title = "Forecast vs Actuals"
                        if item_col != "No filter" and selected_items:
                            title += f" - {len(selected_items)} Items"
                        
                        fig.update_layout(
                            title=title,
                            xaxis_title='Date',
                            yaxis_title='Value',
                            hovermode='x unified',
                            template='plotly_white',
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Plotly not available for advanced visualizations")

                    # Accuracy Metrics with Conditional Formatting
                    if 'y' in combined_result.columns:
                        actuals_with_forecast = combined_result.dropna(subset=['y', 'yhat'])
                        if len(actuals_with_forecast) > 0:
                            st.subheader("🔍 Forecast Accuracy Metrics")
                            
                            # Calculate metrics per item if multiple items
                            if item_col != "No filter" and selected_items:
                                metrics_data = []
                                for item in selected_items:
                                    item_data = actuals_with_forecast[actuals_with_forecast[item_col] == item]
                                    if len(item_data) > 0:
                                        mape = mean_absolute_percentage_error(item_data['y'], item_data['yhat'])
                                        rmse = np.sqrt(mean_squared_error(item_data['y'], item_data['yhat'])) if SKLEARN_AVAILABLE else simple_rmse(item_data['y'], item_data['yhat'])
                                        r2 = r2_score(item_data['y'], item_data['yhat']) if SKLEARN_AVAILABLE else simple_r2(item_data['y'], item_data['yhat'])
                                        
                                        metrics_data.append({
                                            'Item': item,
                                            'MAPE': mape,
                                            'RMSE': rmse,
                                            'R²': r2
                                        })
                                
                                if metrics_data:
                                    metrics_df = pd.DataFrame(metrics_data)
                                    st.dataframe(metrics_df.style.format({
                                        'MAPE': '{:.1f}%',
                                        'RMSE': '{:.2f}',
                                        'R²': '{:.3f}'
                                    }))
                            else:
                                # Single time series metrics
                                mape = mean_absolute_percentage_error(actuals_with_forecast['y'], actuals_with_forecast['yhat'])
                                rmse = np.sqrt(mean_squared_error(actuals_with_forecast['y'], actuals_with_forecast['yhat'])) if SKLEARN_AVAILABLE else simple_rmse(actuals_with_forecast['y'], actuals_with_forecast['yhat'])
                                r2 = r2_score(actuals_with_forecast['y'], actuals_with_forecast['yhat']) if SKLEARN_AVAILABLE else simple_r2(actuals_with_forecast['y'], actuals_with_forecast['yhat'])
                                
                                # Display metrics with conditional formatting
                                cols = st.columns(3)
                                metrics = [
                                    ('MAPE', f"{mape:.1f}%", get_metric_color('mape', mape)),
                                    ('RMSE', f"{rmse:.2f}", get_metric_color('rmse', rmse)),
                                    ('R²', f"{r2:.3f}", get_metric_color('r2', r2))
                                ]
                                
                                for col, (label, value, color) in zip(cols, metrics):
                                    col.markdown(f"""
                                        <div style="
                                            border-left: 4px solid {color};
                                            padding: 12px;
                                            background-color: {color}10;
                                            border-radius: 8px;
                                            margin-bottom: 10px;
                                        ">
                                            <div style="font-size: 0.9em; color: #444; margin-bottom: 8px;">{label}</div>
                                            <div style="font-size: 1.8em; font-weight: bold; color: {color}">{value}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                # Interpretation guide
                                st.markdown(f"""
                                <div style='background: {CARD}; padding: 1rem; border-radius: 8px; border: 1px solid {BORDER}; margin-top: 1rem;'>
                                    <h4 style='color: {PRIMARY}; margin-bottom: 0.5rem;'>📊 Metric Interpretation</h4>
                                    <ul style='margin: 0;'>
                                        <li><strong style='color: {ACCENT};'>MAPE &lt; 10%:</strong> Excellent forecast accuracy</li>
                                        <li><strong style='color: #F59E0B;'>MAPE 10-20%:</strong> Good forecast accuracy</li>
                                        <li><strong style='color: #EF4444;'>MAPE &gt; 20%:</strong> Needs improvement</li>
                                        <li><strong style='color: {ACCENT};'>R² &gt; 0.7:</strong> Strong relationship</li>
                                        <li><strong style='color: #F59E0B;'>R² 0.5-0.7:</strong> Moderate relationship</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Enhanced export with item name
                    st.subheader("💾 Export Forecast")
                    download_df = combined_result.copy()
                    
                    csv = download_df.to_csv(index=False)
                    filename = "supply_chain_forecast"
                    if selected_items:
                        filename += f"_{len(selected_items)}_items"
                    else:
                        filename += "_all_data"
                    filename += f"_{datetime.now().strftime('%Y%m%d')}.csv"
                    
                    st.download_button(
                        "⬇️ Download Enhanced Forecast CSV",
                        csv,
                        filename,
                        "text/csv"
                    )
                    
                    st.success("✅ Forecast generated and ready for analysis!")
                    
                except Exception as e:
                    st.error(f"Forecast failed: {str(e)}")

    else:
        st.info("ℹ️ Please upload your data to begin forecasting")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; padding: 20px 0;'>
        <div style='font-size: 3em; margin-bottom: 10px; color: {PRIMARY};'>📊</div>
        <h2 style='margin-bottom: 5px; color: {TEXT};'>Chris Kimau</h2>
        <div class='badge' style='display: inline-block; margin: 10px 0;'>
            Supply Chain Analytics
        </div>
        <div style='font-size: 0.9em; color: {ACCENT}; margin-top: 5px;'>
            Forecasting & Dashboard Specialist
        </div>
        <div style='font-size: 0.8em; color: {SUBTEXT}; margin-top: 8px;'>
            Skanem Africa · Full-time
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    # Navigation - UPDATED to include Forecasting
    st.markdown("### 🧭 Navigation")
    nav_options = ["🏠 Home", "👨‍💻 Profile", "💼 Experience", "📊 Dashboards", "🔮 Forecasting", "🚀 Projects", "🛠️ Skills", "📞 Contact"]
    selected_nav = st.radio("", nav_options, label_visibility="collapsed")
    
    st.write("---")
    
    # Supply Chain KPIs
    st.markdown("### 📈 Supply Chain KPIs")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Forecast Accuracy", "94%")
        st.metric("Inventory Turnover", "8.2x")
    with col2:
        st.metric("Service Level", "98.5%")
        st.metric("Cost Reduction", "15%")
    
    st.write("---")
    
    # Download CV
    st.markdown("### 📄 Resume")
    st.download_button(
        label="📥 Download CV",
        data=get_cv_bytes(),
        file_name="Chris_Kimau_Supply_Chain_CV.txt",
        mime="text/plain"
    )
    
    st.write("---")
    
    # Social Links
    st.markdown("### 🌐 Connect")
    st.markdown(f"""
    <div style='display: flex; flex-direction: column; gap: 8px;'>
        <a href='https://linkedin.com/in/chrismukitikimau' style='text-decoration: none;'>
            <div style='padding: 12px; background: #0077B5; color: white; border-radius: 8px; text-align: center; font-weight: 500;'>
                💼 LinkedIn Profile
            </div>
        </a>
        <a href='https://github.com/Ckimau' style='text-decoration: none;'>
            <div style='padding: 12px; background: #333; color: white; border-radius: 8px; text-align: center; font-weight: 500;'>
                💻 GitHub Profile
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Main Content based on Navigation
# ---------------------------
if "🏠 Home" in selected_nav:
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style='margin-top: 2rem;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 1.5rem; line-height: 1.2; color: {TEXT};'>
            Transforming Supply Chains with <span style='color: {PRIMARY}'>Data & Analytics</span>
            </h1>
            <div class='readable-text'>
            Supply Chain & Analytics Specialist with 6+ years of expertise in predictive analytics, 
            interactive dashboard development, and logistics planning. I combine advanced data science 
            with business intelligence tools to deliver actionable insights and measurable improvements 
            in supply chain performance.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Core Capabilities
        st.markdown("### 🎯 Core Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='supply-chain-feature'>
                <h4 style='color: {PRIMARY}; margin-bottom: 12px;'>📊 BI Dashboards</h4>
                <p>Interactive Tableau & Power BI dashboards for real-time supply chain monitoring</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='supply-chain-feature'>
                <h4 style='color: {PRIMARY}; margin-bottom: 12px;'>📈 Demand Forecasting</h4>
                <p>AI-powered demand prediction and inventory optimization using machine learning</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='supply-chain-feature'>
                <h4 style='color: {PRIMARY}; margin-bottom: 12px;'>🚚 Logistics Analytics</h4>
                <p>Transportation optimization and distribution network analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 View Dashboards", use_container_width=True):
                st.session_state.nav = "📊 Dashboards"
        with col2:
            if st.button("🔮 Forecasting Demo", use_container_width=True):
                st.session_state.nav = "🔮 Forecasting"
        with col3:
            if st.button("📞 Contact Me", use_container_width=True):
                st.session_state.nav = "📞 Contact"
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem;'>
            <div style='font-size: 8rem; margin-bottom: 1rem; color: {PRIMARY};'>📈</div>
            <div class='badge' style='margin-top: 1rem;'>Available for Projects</div>
            <div style='margin-top: 2rem; padding: 1.5rem; background: rgba(37, 99, 235, 0.08); border-radius: 12px;'>
                <h4 style='color: {PRIMARY}; margin-bottom: 8px;'>Current Role</h4>
                <p style='margin: 0; font-weight: 600; color: {TEXT};'>Supply Chain Manager</p>
                <p style='margin: 4px 0; color: {SUBTEXT};'>Skanem Africa</p>
                <p style='margin: 0; color: {SUBTEXT};'>Oct 2024 - Present</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("---")
    
    # Impact Metrics
    st.markdown("### 📊 Measurable Business Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dashboard Adoption", "95%", "User engagement")
    with col2:
        st.metric("Inventory Reduction", "35%", "Excess stock optimization")
    with col3:
        st.metric("Forecast Accuracy", "+25%", "Through AI models")
    with col4:
        st.metric("Cost Savings", "15%", "Logistics optimization")
        
    # Sample Forecast Chart
    st.markdown("### 📈 Sample Supply Chain Forecast")
    sample_data = create_sample_forecast_data()
    
    # Create Altair chart
    chart_data = sample_data.melt('Month', var_name='Type', value_name='Value')
    forecast_chart = alt.Chart(chart_data).mark_line().encode(
        x='Month:T',
        y='Value:Q',
        color=alt.Color('Type:N', scale=alt.Scale(
            domain=['Actual', 'Forecast'],
            range=[PRIMARY, ACCENT]
        )),
        strokeDash=alt.StrokeDash('Type:N', scale=alt.Scale(
            domain=['Actual', 'Forecast'],
            range=[[0], [5, 5]]
        ))
    ).properties(
        height=300,
        title='Monthly Demand Forecast vs Actual'
    )
    
    st.altair_chart(forecast_chart, use_container_width=True)
    
    # Show forecast metrics using custom functions (no sklearn dependency)
    if len(sample_data) > 0:
        st.markdown("### 📋 Forecast Performance")
        actual_values = sample_data['Actual'].values
        forecast_values = sample_data['Forecast'].values
        
        col1, col2, col3 = st.columns(3)
        with col1:
            mape = safe_mape(actual_values, forecast_values)
            st.metric("MAPE", f"{mape:.1f}%")
        with col2:
            rmse = simple_rmse(actual_values, forecast_values)
            st.metric("RMSE", f"{rmse:.0f}")
        with col3:
            r2 = simple_r2(actual_values, forecast_values)
            st.metric("R² Score", f"{r2:.3f}")

elif "👨‍💻 Profile" in selected_nav:
    st.markdown("## 👨‍💻 Professional Profile")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class='neon-card'>
            <h3 style='color: {PRIMARY}; margin-bottom: 1.5rem;'>Supply Chain Analytics Specialist</h3>
            <div class='readable-text'>
            I am a results-driven Supply Chain professional specializing in data analytics, dashboard development, 
            and predictive modeling. With extensive experience in manufacturing and distribution environments, 
            I bridge the gap between operational excellence and data-driven decision making through interactive 
            visualizations and advanced analytics.
            </div>
            
            <div class='readable-text'>
            My expertise lies in creating comprehensive Tableau and Power BI dashboards that transform complex 
            supply chain data into actionable insights. I have successfully delivered projects that significantly 
            improve forecast accuracy, reduce inventory costs, and enhance overall supply chain visibility.
            </div>
            
            <h4 style='color: {PRIMARY}; margin-top: 2rem; margin-bottom: 1rem;'>Key Focus Areas:</h4>
            <ul>
            <li><strong>Dashboard Development:</strong> Creating interactive Tableau and Power BI dashboards for real-time monitoring</li>
            <li><strong>Demand Planning:</strong> Developing accurate forecasting models using time series analysis</li>
            <li><strong>Inventory Strategy:</strong> Optimizing stock levels through data-driven insights</li>
            <li><strong>Logistics Optimization:</strong> Designing efficient distribution networks</li>
            <li><strong>Process Improvement:</strong> Implementing data-driven approaches to enhance efficiency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='neon-card'>
            <h3 style='color: {PRIMARY}; margin-bottom: 1.5rem;'>🏆 Key Achievements</h3>
            <div class='achievement-card'>
                <h4 style='color: {ACCENT}; margin: 0;'>95%</h4>
                <p style='margin: 8px 0 0 0;'>Dashboard Adoption Rate</p>
                <small>Across supply chain teams</small>
            </div>
            <div class='achievement-card'>
                <h4 style='color: {ACCENT}; margin: 0;'>35%</h4>
                <p style='margin: 8px 0 0 0;'>Excess Inventory Reduction</p>
                <small>Through optimization dashboards</small>
            </div>
            <div class='achievement-card'>
                <h4 style='color: {ACCENT}; margin: 0;'>25%</h4>
                <p style='margin: 8px 0 0 0;'>Forecast Accuracy Improvement</p>
                <small>AI-driven models</small>
            </div>
            <div class='achievement-card'>
                <h4 style='color: {ACCENT}; margin: 0;'>15%</h4>
                <p style='margin: 8px 0 0 0;'>Logistics Cost Reduction</p>
                <small>Route optimization dashboards</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # BI Tools Expertise
    st.markdown("## 🛠️ Business Intelligence Expertise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>📊 Tableau Specialization</h4>
            <ul>
            <li>Interactive supply chain dashboards</li>
            <li>Real-time KPI monitoring</li>
            <li>Advanced data blending</li>
            <li>Parameter controls and filters</li>
            <li>Dashboard performance optimization</li>
            <li>Server administration and publishing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>💡 Power BI Expertise</h4>
            <ul>
            <li>DAX formula development</li>
            <li>Data modeling and relationships</li>
            <li>Power Query transformations</li>
            <li>Custom visualizations</li>
            <li>Report publishing and sharing</li>
            <li>Automated data refresh</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif "📊 Dashboards" in selected_nav:
    st.markdown("## 📊 Tableau & Power BI Dashboards")
    
    st.markdown(f"""
    <div class='neon-card'>
        <h3 style='color: {PRIMARY}; margin-bottom: 1rem;'>Interactive Supply Chain Dashboards</h3>
        <div class='readable-text'>
        I specialize in creating comprehensive Tableau and Power BI dashboards that transform complex supply chain data 
        into actionable insights. Below are examples of dashboard solutions I've developed for inventory management, 
        demand forecasting, and logistics optimization.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard Examples
    st.markdown("### 🎯 Dashboard Portfolio")
    
    # Dashboard 1: Inventory Management
    st.markdown(f"""
    <div class='dashboard-preview'>
        <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>📦 Inventory Optimization Dashboard</h4>
        <div class='readable-text'>
        Real-time inventory tracking with stockout risk analysis, turnover rates, and replenishment recommendations.
        </div>
        
        <div style='margin: 1.5rem 0;'>
            <strong>Key Features:</strong>
            <ul>
            <li>SKU-level inventory tracking</li>
            <li>Stockout risk scoring</li>
            <li>Turnover rate analysis</li>
            <li>Automated reorder alerts</li>
            <li>Supplier performance metrics</li>
            </ul>
        </div>
        
        <div style='background: linear-gradient(135deg, {PRIMARY}20, {ACCENT}20); padding: 2rem; border-radius: 8px; text-align: center; margin: 1rem 0;'>
            <div style='font-size: 3rem; color: {PRIMARY};'>📊</div>
            <p style='margin: 1rem 0 0 0; color: {SUBTEXT};'>Interactive Tableau Dashboard</p>
        </div>
        
        <div style='display: flex; gap: 1rem; margin-top: 1.5rem;'>
            <div style='flex: 1; text-align: center; padding: 1rem; background: rgba(37, 99, 235, 0.08); border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: {PRIMARY};'>35%</div>
                <div style='font-size: 0.9rem; color: {SUBTEXT};'>Inventory Reduction</div>
            </div>
            <div style='flex: 1; text-align: center; padding: 1rem; background: rgba(5, 150, 105, 0.08); border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: {ACCENT};'>20%</div>
                <div style='font-size: 0.9rem; color: {SUBTEXT};'>Stockout Reduction</div>
            </div>
            <div style='flex: 1; text-align: center; padding: 1rem; background: rgba(124, 58, 237, 0.08); border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: {SECONDARY};'>95%</div>
                <div style='font-size: 0.9rem; color: {SUBTEXT};'>User Adoption</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard 2: Demand Forecasting
    st.markdown(f"""
    <div class='dashboard-preview'>
        <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>📈 Demand Planning Dashboard</h4>
        <div class='readable-text'>
        Advanced forecasting dashboard with machine learning integration, seasonality analysis, and accuracy tracking.
        </div>
        
        <div style='margin: 1.5rem 0;'>
            <strong>Key Features:</strong>
            <ul>
            <li>Machine learning forecasts</li>
            <li>Seasonality pattern analysis</li>
            <li>Forecast accuracy tracking</li>
            <li>Scenario planning tools</li>
            <li>Executive summary views</li>
            </ul>
        </div>
        
        <div style='background: linear-gradient(135deg, {ACCENT}20, {SECONDARY}20); padding: 2rem; border-radius: 8px; text-align: center; margin: 1rem 0;'>
            <div style='font-size: 3rem; color: {ACCENT};'>🔮</div>
            <p style='margin: 1rem 0 0 0; color: {SUBTEXT};'>Power BI Forecasting Suite</p>
        </div>
        
        <div style='display: flex; gap: 1rem; margin-top: 1.5rem;'>
            <div style='flex: 1; text-align: center; padding: 1rem; background: rgba(37, 99, 235, 0.08); border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: {PRIMARY};'>94%</div>
                <div style='font-size: 0.9rem; color: {SUBTEXT};'>Forecast Accuracy</div>
            </div>
            <div style='flex: 1; text-align: center; padding: 1rem; background: rgba(5, 150, 105, 0.08); border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: {ACCENT};'>25%</div>
                <div style='font-size: 0.9rem; color: {SUBTEXT};'>Accuracy Improvement</div>
            </div>
            <div style='flex: 1; text-align: center; padding: 1rem; background: rgba(124, 58, 237, 0.08); border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: {SECONDARY};'>15%</div>
                <div style='font-size: 0.9rem; color: {SUBTEXT};'>Revenue Growth</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard 3: Logistics Analytics
    st.markdown(f"""
    <div class='dashboard-preview'>
        <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>🚚 Logistics Optimization Dashboard</h4>
        <div class='readable-text'>
        Comprehensive logistics monitoring with route optimization, carrier performance, and cost analysis.
        </div>
        
        <div style='margin: 1.5rem 0;'>
            <strong>Key Features:</strong>
            <ul>
            <li>Route optimization analysis</li>
            <li>Carrier performance scoring</li>
            <li>Fuel consumption tracking</li>
            <li>Delivery time analysis</li>
            <li>Cost per mile metrics</li>
            </ul>
        </div>
        
        <div style='background: linear-gradient(135deg, {SECONDARY}20, {PRIMARY}20); padding: 2rem; border-radius: 8px; text-align: center; margin: 1rem 0;'>
            <div style='font-size: 3rem; color: {SECONDARY};'>📋</div>
            <p style='margin: 1rem 0 0 0; color: {SUBTEXT};'>Tableau Logistics Suite</p>
        </div>
        
        <div style='display: flex; gap: 1rem; margin-top: 1.5rem;'>
            <div style='flex: 1; text-align: center; padding: 1rem; background: rgba(37, 99, 235, 0.08); border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: {PRIMARY};'>15%</div>
                <div style='font-size: 0.9rem; color: {SUBTEXT};'>Cost Reduction</div>
            </div>
            <div style='flex: 1; text-align: center; padding: 1rem; background: rgba(5, 150, 105, 0.08); border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: {ACCENT};'>98.5%</div>
                <div style='font-size: 0.9rem; color: {SUBTEXT};'>On-time Delivery</div>
            </div>
            <div style='flex: 1; text-align: center; padding: 1rem; background: rgba(124, 58, 237, 0.08); border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; color: {SECONDARY};'>22%</div>
                <div style='font-size: 0.9rem; color: {SUBTEXT};'>Route Efficiency</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample Dashboard Data Table
    st.markdown("### 📋 Sample Dashboard Metrics")
    
    # Create sample data
    inventory_data = pd.DataFrame({
        'Product': ['BOPP 35µ Film', 'White PE', 'BOPP 20µ Film', 'Clear PP', 'Metallized Film'],
        'Current Stock': [1250, 890, 1100, 750, 600],
        'Safety Stock': [500, 300, 400, 250, 200],
        'Monthly Demand': [1500, 1000, 1200, 800, 500],
        'Stockout Risk': ['Low', 'Medium', 'Low', 'High', 'Medium']
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Inventory Overview")
        st.dataframe(inventory_data, use_container_width=True)
    
    with col2:
        st.markdown("#### Performance Metrics")
        metrics_data = pd.DataFrame({
            'Metric': ['Forecast Accuracy', 'Inventory Turnover', 'Service Level', 'Cost Reduction'],
            'Current': ['94%', '8.2x', '98.5%', '15%'],
            'Target': ['95%', '9.0x', '99%', '20%'],
            'Status': ['On Track', 'Improving', 'Excellent', 'Good']
        })
        st.dataframe(metrics_data, use_container_width=True)

# NEW FORECASTING SECTION
elif "🔮 Forecasting" in selected_nav:
    forecasting_app()

elif "🚀 Projects" in selected_nav:
    st.markdown("## 🚀 Supply Chain Projects")
    
    projects = [
        {
            'title': 'Tableau Supply Chain Dashboard Suite',
            'description': 'Comprehensive Tableau dashboard suite for end-to-end supply chain visibility including inventory management, demand forecasting, and logistics optimization.',
            'technologies': ['Tableau', 'SQL', 'Python', 'SAP Integration'],
            'impact': 'Improved decision-making speed by 40% and reduced inventory costs by 35% through real-time analytics',
            'status': '🚀 Production',
            'category': 'Business Intelligence'
        },
        {
            'title': 'Power BI Forecasting Platform',
            'description': 'Advanced Power BI platform integrating machine learning forecasts with interactive dashboards for demand planning and inventory optimization.',
            'technologies': ['Power BI', 'Python', 'Machine Learning', 'DAX'],
            'impact': 'Achieved 94% forecast accuracy and 25% improvement in planning efficiency',
            'status': '🚀 Production', 
            'category': 'Analytics Platform'
        },
        {
            'title': 'Inventory Optimization Dashboard',
            'description': 'Real-time inventory tracking dashboard with stockout risk analysis and automated replenishment recommendations.',
            'technologies': ['Tableau', 'SQL', 'Automation'],
            'impact': 'Reduced stockouts by 20% and excess inventory by 35% through predictive analytics',
            'status': '🚀 Production',
            'category': 'Inventory Management'
        }
    ]
    
    for i, project in enumerate(projects):
        with st.container():
            st.markdown(f"""
            <div class='neon-card'>
                <div style='display: flex; justify-content: between; align-items: start; margin-bottom: 1.5rem;'>
                    <div>
                        <h3 style='color: {PRIMARY}; margin-bottom: 0.5rem;'>{project['title']}</h3>
                        <span class='tech-tag'>{project['category']}</span>
                    </div>
                    <span class='badge'>{project['status']}</span>
                </div>
                <div class='readable-text'>{project['description']}</div>
                <p style='margin: 1rem 0;'><strong>📈 Business Impact:</strong> {project['impact']}</p>
                <div style='margin: 1.5rem 0;'>
            """, unsafe_allow_html=True)
            
            for tech in project['technologies']:
                st.markdown(f"<span class='tech-tag'>{tech}</span>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.write("---")

elif "🛠️ Skills" in selected_nav:
    st.markdown("## 🛠️ Supply Chain & Analytics Expertise")
    
    # Skills Visualization
    st.altair_chart(create_supply_chain_skill_chart(), use_container_width=True)
    
    # Skills Categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>📊 Business Intelligence</h4>
            <ul>
            <li>Tableau Dashboard Development</li>
            <li>Power BI Reporting</li>
            <li>Data Visualization</li>
            <li>KPI Design</li>
            <li>Executive Reporting</li>
            <li>Interactive Dashboards</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>🤖 Data Science & Analytics</h4>
            <ul>
            <li>Machine Learning</li>
            <li>Statistical Analysis</li>
            <li>Time Series Forecasting</li>
            <li>Predictive Modeling</li>
            <li>Python Programming</li>
            <li>SQL Database Management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>🏭 Supply Chain Management</h4>
            <ul>
            <li>Demand Forecasting</li>
            <li>Inventory Optimization</li>
            <li>Logistics Planning</li>
            <li>SAP Systems</li>
            <li>Process Improvement</li>
            <li>Supplier Management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif "💼 Experience" in selected_nav:
    st.markdown("## 💼 Professional Experience")
    
    # Current Role - Highlighted
    st.markdown(f"""
    <div class='role-highlight'>
        <div style='display: flex; justify-content: between; align-items: start;'>
            <div>
                <h3 style='margin: 0; color: {PRIMARY};'>Warehouse & Inventory Manager</h3>
                <h4 style='margin: 8px 0; color: {TEXT};'>Skanem Africa · Full-time</h4>
                <p style='margin: 0; color: {SUBTEXT};'>Oct 2024 - Present</p>
            </div>
            <span class='badge'>Current Role</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='neon-card' style='margin-top: 0;'>
        <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>Supply Chain & Analytics Responsibilities:</h4>
        <ul>
        <li>Implement demand forecasting pipelines and inventory optimization dashboards</li>
        <li>Develop Tableau and Power BI reports for supply chain performance monitoring</li>
        <li>Oversee finished goods inventory and SKU-level tracking systems</li>
        <li>Collaborate with logistics partners on distribution planning and optimization</li>
        <li>Lead process improvement initiatives to enhance supply chain efficiency</li>
        <li>Manage safety stock levels and reorder point calculations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Previous Roles
    experiences = [
        {
            'title': 'Supply Chain Analyst',
            'company': 'Mabati Rolling Mills',
            'period': 'Jan 2024 – Oct 2024',
            'achievements': [
                'Developed AI-driven demand forecasting models reducing stockouts by 20%',
                'Created interactive Power BI dashboards improving sales efficiency by 15%',
                'Optimized inventory levels, reducing excess stock by 35% while maintaining service levels',
                'Collaborated with cross-functional teams to enhance data-driven decision-making',
                'Implemented supply chain performance metrics and reporting frameworks'
            ]
        },
        {
            'title': 'Warehouse Officer',
            'company': 'Mabati Rolling Mills', 
            'period': 'July 2022 – 2023',
            'achievements': [
                'Managed data-driven forecasting for supply chain continuity and risk mitigation',
                'Improved inventory accuracy by 18% through process optimization and system enhancements',
                'Integrated machine learning models for demand forecasting and lead time reduction',
                'Optimized warehouse layout and storage strategies for improved efficiency'
            ]
        }
    ]
    
    for exp in experiences:
        st.markdown(f"""
        <div class='neon-card'>
            <h3 style='color: {PRIMARY}; margin-bottom: 8px;'>{exp['title']}</h3>
            <h4 style='margin: 4px 0; color: {TEXT};'>{exp['company']}</h4>
            <p style='margin: 0 0 1rem 0; color: {SUBTEXT};'>{exp['period']}</p>
            <ul style='margin-top: 1rem;'>
        """, unsafe_allow_html=True)
        
        for achievement in exp['achievements']:
            st.markdown(f"<li class='readable-text'>{achievement}</li>", unsafe_allow_html=True)
            
        st.markdown("</ul></div>", unsafe_allow_html=True)

elif "📞 Contact" in selected_nav:
    st.markdown("## 📞 Get In Touch")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class='neon-card'>
            <h3 style='color: {PRIMARY}; margin-bottom: 1.5rem;'>Let's Transform Your Supply Chain</h3>
            <div class='readable-text'>
            I'm passionate about helping organizations optimize their supply chain operations through data-driven 
            forecasting and planning. Whether you're looking to improve forecast accuracy, reduce inventory costs, 
            or optimize logistics operations, I can help you achieve measurable results.
            </div>
            
            <h4 style='color: {PRIMARY}; margin-top: 2rem; margin-bottom: 1rem;'>How I Can Help:</h4>
            <ul>
            <li><strong>Demand Forecasting:</strong> Implement AI-driven forecasting models for better accuracy</li>
            <li><strong>Inventory Optimization:</strong> Reduce costs while maintaining service levels</li>
            <li><strong>Supply Chain Analytics:</strong> Build dashboards and reporting systems</li>
            <li><strong>Process Improvement:</strong> Streamline supply chain operations</li>
            <li><strong>Technology Implementation:</strong> Deploy supply chain management systems</li>
            <li><strong>Logistics Optimization:</strong> Design efficient distribution networks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1.5rem;'>👤 Contact Information</h4>
            <div style='margin-bottom: 1.5rem;'>
                <h5 style='color: {TEXT}; margin-bottom: 8px;'>📍 Location</h5>
                <p style='margin: 0; color: {SUBTEXT};'>Nairobi, Kenya</p>
            </div>
            <div style='margin-bottom: 1.5rem;'>
                <h5 style='color: {TEXT}; margin-bottom: 8px;'>📧 Email</h5>
                <p style='margin: 0; color: {SUBTEXT};'>kimauchris0@gmail.com</p>
            </div>
            <div style='margin-bottom: 2rem;'>
                <h5 style='color: {TEXT}; margin-bottom: 8px;'>📱 Phone</h5>
                <p style='margin: 0; color: {SUBTEXT};'>+254 706 109 248</p>
            </div>
            
            <div style='margin-top: 2rem;'>
                <a href='https://linkedin.com/in/chrismukitikimau' style='text-decoration: none;'>
                    <div style='padding: 12px; background: #0077B5; color: white; border-radius: 8px; text-align: center; margin: 8px 0; font-weight: 500;'>
                        💼 LinkedIn Profile
                    </div>
                </a>
                <a href='https://github.com/Ckimau' style='text-decoration: none;'>
                    <div style='padding: 12px; background: #333; color: white; border-radius: 8px; text-align: center; margin: 8px 0; font-weight: 500;'>
                        💻 GitHub Profile
                    </div>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Contact Form
    st.markdown("### 📝 Send a Message")
    
    with st.form("contact_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
        
        with col2:
            company = st.text_input("Company")
            subject = st.selectbox("Subject", [
                "Supply Chain Consulting",
                "Demand Forecasting", 
                "Inventory Optimization",
                "Logistics Planning",
                "Job Opportunity",
                "Project Collaboration",
                "Other"
            ])
        
        message = st.text_area("Message", height=150, placeholder="Tell me about your supply chain challenges or project requirements...")
        
        submitted = st.form_submit_button("🚀 Send Message")
        
        if submitted:
            if name and email and message:
                st.success("✅ Thank you for your message! I'll get back to you within 24 hours.")
            else:
                st.warning("⚠️ Please fill in all required fields.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {SUBTEXT}; padding: 2rem 0;'>
    <p style='margin-bottom: 0.5rem;'>Built with ❤️ using Streamlit • {datetime.now().year} Chris Kimau</p>
    <p style='margin: 0; font-size: 0.9em;'>Transforming supply chains through data analytics and business intelligence</p>
</div>
""", unsafe_allow_html=True)
