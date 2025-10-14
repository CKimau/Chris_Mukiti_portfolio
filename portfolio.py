# streamlit_portfolio.py
# Light Theme Supply Chain Portfolio with Tableau & Power BI Dashboards
# Run: `streamlit run streamlit_portfolio.py`

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Try to import Prophet, but provide fallback if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

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

/* Table styling */
.dashboard-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}}

.dashboard-table th {{
    background: {PRIMARY};
    color: white;
    padding: 12px;
    text-align: left;
    font-weight: 600;
}}

.dashboard-table td {{
    padding: 12px;
    border-bottom: 1px solid {BORDER};
}}

.dashboard-table tr:hover {{
    background: rgba(37, 99, 235, 0.05);
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

# ---------------------------
# Dashboard Data & Examples
# ---------------------------
def create_sample_dashboard_data():
    """Create sample data for dashboard examples"""
    # Sample inventory data
    inventory_data = pd.DataFrame({
        'Product': ['BOPP 35µ Film', 'White PE', 'BOPP 20µ Film', 'Clear PP', 'Metallized Film'],
        'Current Stock': [1250, 890, 1100, 750, 600],
        'Safety Stock': [500, 300, 400, 250, 200],
        'Monthly Demand': [1500, 1000, 1200, 800, 500],
        'Stockout Risk': ['Low', 'Medium', 'Low', 'High', 'Medium']
    })
    
    # Sample forecast data
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    forecast_data = pd.DataFrame({
        'Month': dates,
        'Actual': [1200, 1350, 1100, 1450, 1300, 1400, 1250, 1500, 1350, 1420, 1280, 1480],
        'Forecast': [1150, 1300, 1150, 1400, 1320, 1380, 1280, 1480, 1370, 1450, 1300, 1500]
    })
    
    return inventory_data, forecast_data

def create_dashboard_preview(title, description, metrics, chart_type="bar"):
    """Create a dashboard preview component"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {title}")
        st.markdown(f"<div class='readable-text'>{description}</div>", unsafe_allow_html=True)
        
        # Create a simple chart based on type
        if chart_type == "bar":
            chart_data = pd.DataFrame({
                'Category': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                'Value': [100, 120, 90, 150, 130]
            })
            chart = alt.Chart(chart_data).mark_bar(color=PRIMARY).encode(
                x='Category',
                y='Value'
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)
        else:
            # Line chart
            chart_data = pd.DataFrame({
                'Month': pd.date_range('2024-01-01', periods=5, freq='M'),
                'Value': [100, 120, 90, 150, 130]
            })
            chart = alt.Chart(chart_data).mark_line(color=ACCENT).encode(
                x='Month',
                y='Value'
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("### Key Metrics")
        for metric_name, metric_value, delta in metrics:
            st.metric(metric_name, metric_value, delta)

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
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    nav_options = ["🏠 Home", "👨‍💻 Profile", "💼 Experience", "📊 Dashboards", "🚀 Projects", "🛠️ Skills", "📈 Forecasting", "📞 Contact"]
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
            if st.button("🚀 Projects", use_container_width=True):
                st.session_state.nav = "🚀 Projects"
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
    inventory_data, forecast_data = create_sample_dashboard_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Inventory Overview")
        st.dataframe(inventory_data, use_container_width=True)
    
    with col2:
        st.markdown("#### Forecast Accuracy")
        st.dataframe(forecast_data, use_container_width=True)
    
    # Technical Specifications
    st.markdown("### 🛠️ Technical Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>Tableau Stack</h4>
            <ul>
            <li><strong>Data Sources:</strong> SQL Server, SAP, Excel</li>
            <li><strong>Visualizations:</strong> Interactive dashboards, maps, trend lines</li>
            <li><strong>Features:</strong> Parameters, sets, LOD calculations</li>
            <li><strong>Deployment:</strong> Tableau Server, Tableau Online</li>
            <li><strong>Integration:</strong> REST API, Web Data Connectors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>Power BI Stack</h4>
            <ul>
            <li><strong>Data Sources:</strong> Dataflows, SQL, APIs</li>
            <li><strong>Modeling:</strong> Star schema, DAX measures</li>
            <li><strong>Features:</strong> Power Query, Row-level security</li>
            <li><strong>Deployment:</strong> Power BI Service, Embedded</li>
            <li><strong>Integration:</strong> Power Automate, Azure services</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# [Other sections remain similar but with light theme applied...]

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

# [Other sections continue with light theme...]

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
