# streamlit_portfolio.py
# Enhanced Supply Chain & Forecasting Portfolio for Chris Kimau
# Run: `streamlit run streamlit_portfolio.py`

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import sqlite3
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# ---------------------------
# Page config & Theme
# ---------------------------
st.set_page_config(
    page_title="Chris Kimau — Supply Chain Forecasting & Demand Planning Specialist",
    page_icon="", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme with better contrast
PRIMARY = "#2563EB"           # Professional blue
ACCENT = "#059669"            # Professional green
SECONDARY = "#7C3AED"         # Purple
BG = "#0F172A"                # Dark blue
CARD = "#1E293B"
TEXT = "#01595C"              # Light text for better readability
SUBTEXT = "#FF7B00"           # Lighter subtext

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
}}

* {{
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    line-height: 1.6;
}}

html, body, [class*="css"] {{
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: var(--text);
    background-attachment: fixed;
}}

h1, h2, h3, h4, h5, h6 {{
    color: var(--text) !important;
    font-weight: 600;
    line-height: 1.3;
}}

p, li, div {{
    color: var(--subtext) !important;
    font-size: 1.05rem;
    line-height: 1.7;
}}

header[data-testid="stHeader"] {{
    background: rgba(15, 23, 42, 0.9);
    backdrop-filter: blur(10px);
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

.glow-text {{
    text-shadow: 0 0 20px var(--primary), 0 0 40px rgba(37, 99, 235, 0.3);
}}

.neon-card {{
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(37, 99, 235, 0.3);
    border-radius: 16px;
    padding: 28px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    margin-bottom: 24px;
}}

.neon-card:hover {{
    border-color: var(--primary);
    box-shadow: 0 12px 40px rgba(37, 99, 235, 0.2);
    transform: translateY(-4px);
}}

.stButton>button {{
    background: linear-gradient(90deg, var(--primary), #1D4ED8);
    color: white;
    border: none;
    padding: 14px 28px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
}}

.stButton>button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
}}

.badge {{
    background: linear-gradient(90deg, var(--accent), #047857);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9em;
}}

.tech-tag {{
    background: rgba(37, 99, 235, 0.15);
    color: var(--primary);
    padding: 6px 14px;
    border-radius: 12px;
    border: 1px solid rgba(37, 99, 235, 0.3);
    font-size: 0.9em;
    margin: 4px;
    font-weight: 500;
}}

.achievement-card {{
    background: rgba(5, 150, 105, 0.1);
    border: 1px solid rgba(5, 150, 105, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
}}

.role-highlight {{
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.15), rgba(37, 99, 235, 0.05));
    border-left: 4px solid var(--primary);
    padding: 20px;
    border-radius: 12px;
    margin: 20px 0;
}}

.supply-chain-feature {{
    background: rgba(30, 41, 59, 0.7);
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
    border: 1px solid rgba(37, 99, 235, 0.2);
}}

/* Hide Streamlit default elements */
footer {{visibility: hidden;}}
#MainMenu {{visibility: hidden;}}

/* Improve readability for dataframes */
.stDataFrame {{
    border-radius: 8px;
    background: rgba(30, 41, 59, 0.5);
}}

/* Better contrast for metrics */
[data-testid="metric-container"] {{
    background: rgba(30, 41, 59, 0.7);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(37, 99, 235, 0.2);
}}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper functions
# ---------------------------
def get_cv_bytes():
    cv_content = """CHRIS KIMAU
Supply Chain Forecasting & Demand Planning Specialist

CONTACT
Mobile: +254706109248
Email: kimauchris0@gmail.com
LinkedIn: linkedin.com/in/chrismukitikimau

PROFESSIONAL SUMMARY
Supply Chain and Data Science professional with 6+ years of expertise in demand forecasting, inventory optimization, and logistics planning. Proven track record of implementing AI-driven forecasting solutions that reduce costs, improve service levels, and optimize supply chain operations across manufacturing and distribution environments.

CORE COMPETENCIES
• Demand Forecasting & Planning
• Inventory Optimization
• Supply Chain Analytics
• Logistics & Distribution
• Machine Learning & AI
• SAP Systems (MM, PP, SD)
• Data Visualization
• Process Improvement

TECHNICAL SKILLS
• Programming: Python, SQL, R
• Machine Learning: Prophet, ARIMA, Scikit-learn, TensorFlow
• Data Visualization: Power BI, Tableau, Streamlit, Plotly
• Supply Chain Systems: SAP, Inventory Management, Demand Planning
• Tools: Git, AWS, Docker, Advanced Excel

EDUCATION & CERTIFICATIONS
• Moringa School: Data Science, Machine Learning & AI
• IBM: Data Science Professional Certificate
• The Catholic University of Eastern Africa: Bachelor of Commerce

PROFESSIONAL EXPERIENCE

Warehouse & Inventory Manager | Skanem Africa (Oct 2024-Present)
• Implement demand forecasting pipelines for inventory optimization
• Oversee finished goods inventory and SKU-level tracking
• Collaborate with supply chain partners on logistics planning
• Lead process improvements for supply chain efficiency

Supply Chain Analyst | Mabati Rolling Mills (Jan 2024-Oct 2024)
• Developed AI-driven demand forecasting models reducing stockouts by 20%
• Created predictive dashboards improving sales efficiency by 15%
• Optimized inventory levels, reducing excess stock by 35%
• Enhanced data-driven decision making across supply chain functions

Warehouse Officer | Mabati Rolling Mills (2022-2023)
• Managed data-driven forecasting for supply chain continuity
• Improved inventory accuracy by 18% through process optimization
• Integrated machine learning models for demand forecasting

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
        'Inventory Optimization': 90,
        'Supply Chain Analytics': 88,
        'Logistics Planning': 85,
        'Machine Learning': 87,
        'SAP Systems': 84,
        'Data Visualization': 86,
        'Process Improvement': 83
    }
    
    chart_data = pd.DataFrame({
        'Skill': list(skills.keys()),
        'Level': list(skills.values())
    })
    
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Level:Q', title='Proficiency Level', scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('Skill:N', title='', sort='-x'),
        color=alt.Color('Level:Q', scale=alt.Scale(range=[PRIMARY, ACCENT]), legend=None)
    ).properties(height=400, title='Supply Chain & Technical Skills')
    
    return chart

def sample_forecast_data():
    """Generate sample forecast data for demonstration"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    actual = np.random.normal(1000, 200, len(dates))
    forecast = actual + np.random.normal(0, 50, len(dates))
    
    return pd.DataFrame({
        'Month': dates,
        'Actual': actual,
        'Forecast': forecast
    })

# ---------------------------
# SForecast App Functions
# ---------------------------
def init_sforecast_db():
    conn = sqlite3.connect("sforecast_demo.db")
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        forecast_type TEXT,
        horizon TEXT,
        forecast_data TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def run_prophet_forecast(df, periods=30):
    """Run Prophet forecasting on the provided data"""
    try:
        m = Prophet()
        m.fit(df.rename(columns={'ds': 'ds', 'y': 'y'}))
        
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        
        return forecast, m
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        return None, None

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; padding: 20px 0;'>
        <div style='font-size: 3em; margin-bottom: 10px;'>📊</div>
        <h2 style='margin-bottom: 5px; color: {TEXT};'>Chris Kimau</h2>
        <div class='badge' style='display: inline-block; margin: 10px 0;'>
            Supply Chain Forecasting Specialist
        </div>
        <div style='font-size: 0.9em; color: {ACCENT}; margin-top: 5px;'>
            Demand Planning & Analytics
        </div>
        <div style='font-size: 0.8em; color: {SUBTEXT}; margin-top: 8px;'>
            Skanem Africa · Full-time
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    nav_options = ["🏠 Home", "👨‍💻 Profile", "💼 Experience", "🚀 Projects", "🛠️ Skills", "📊 Forecasting Demo", "📞 Contact"]
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
            <h1 class='glow-text' style='font-size: 3.5rem; margin-bottom: 1.5rem; line-height: 1.2;'>
            Transforming Supply Chains with <span style='color: {PRIMARY}'>AI-Driven Forecasting</span>
            </h1>
            <div class='readable-text'>
            Supply Chain & Demand Planning Specialist with 6+ years of expertise in predictive analytics, 
            inventory optimization, and logistics planning. I combine advanced data science with supply chain 
            domain knowledge to deliver measurable improvements in forecast accuracy, cost reduction, and 
            operational efficiency.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Supply Chain Focus Areas
        st.markdown("### 🎯 Core Supply Chain Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='supply-chain-feature'>
                <h4 style='color: {PRIMARY}; margin-bottom: 12px;'>📈 Demand Forecasting</h4>
                <p>AI-powered demand prediction using Prophet, ARIMA, and machine learning for accurate inventory planning</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='supply-chain-feature'>
                <h4 style='color: {PRIMARY}; margin-bottom: 12px;'>📦 Inventory Optimization</h4>
                <p>Safety stock optimization, reorder point calculation, and inventory turnover improvement strategies</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='supply-chain-feature'>
                <h4 style='color: {PRIMARY}; margin-bottom: 12px;'>🚚 Logistics Planning</h4>
                <p>Transportation optimization, distribution network design, and cost-effective logistics solutions</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 View Forecasting Demo", use_container_width=True):
                st.session_state.nav = "📊 Forecasting Demo"
        with col2:
            if st.button("💼 Experience", use_container_width=True):
                st.session_state.nav = "💼 Experience"
        with col3:
            if st.button("📞 Contact Me", use_container_width=True):
                st.session_state.nav = "📞 Contact"
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem;'>
            <div style='font-size: 8rem; margin-bottom: 1rem;'>🌐</div>
            <div class='badge' style='margin-top: 1rem;'>Available for Projects</div>
            <div style='margin-top: 2rem; padding: 1.5rem; background: rgba(37, 99, 235, 0.1); border-radius: 12px;'>
                <h4 style='color: {PRIMARY}; margin-bottom: 8px;'>Current Role</h4>
                <p style='margin: 0; font-weight: 600;'>Warehouse & Inventory  Manager</p>
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
        st.metric("Forecast Accuracy Improvement", "+25%", "Through AI models")
    with col2:
        st.metric("Inventory Reduction", "35%", "Excess stock optimization")
    with col3:
        st.metric("Stockout Reduction", "20%", "Predictive tracking")
    with col4:
        st.metric("Cost Savings", "15%", "Logistics optimization")

elif "👨‍💻 Profile" in selected_nav:
    st.markdown("## 👨‍💻 Professional Profile")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class='neon-card'>
            <h3 style='color: {PRIMARY}; margin-bottom: 1.5rem;'>Supply Chain Forecasting Specialist</h3>
            <div class='readable-text'>
            I am a results-driven Supply Chain professional specializing in demand forecasting, inventory optimization, 
            and logistics planning. With extensive experience in manufacturing and distribution environments, I bridge 
            the gap between operational excellence and data-driven decision making.
            </div>
            
            <div class='readable-text'>
            My expertise lies in implementing AI and machine learning solutions that transform traditional supply chain 
            operations into predictive, responsive systems. I have successfully delivered projects that significantly 
            improve forecast accuracy, reduce inventory costs, and enhance overall supply chain performance.
            </div>
            
            <h4 style='color: {PRIMARY}; margin-top: 2rem; margin-bottom: 1rem;'>Key Focus Areas:</h4>
            <ul>
            <li><strong>Demand Planning:</strong> Developing accurate forecasting models using time series analysis and machine learning</li>
            <li><strong>Inventory Strategy:</strong> Optimizing stock levels across complex supply chain networks</li>
            <li><strong>Logistics Optimization:</strong> Designing efficient distribution and transportation solutions</li>
            <li><strong>Process Improvement:</strong> Implementing data-driven approaches to enhance supply chain efficiency</li>
            <li><strong>Technology Integration:</strong> Leveraging AI and analytics platforms for supply chain transformation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='neon-card'>
            <h3 style='color: {PRIMARY}; margin-bottom: 1.5rem;'>🏆 Key Achievements</h3>
            <div class='achievement-card'>
                <h4 style='color: {ACCENT}; margin: 0;'>25%</h4>
                <p style='margin: 8px 0 0 0;'>Forecast Accuracy Improvement</p>
                <small>Through AI-driven models</small>
            </div>
            <div class='achievement-card'>
                <h4 style='color: {ACCENT}; margin: 0;'>35%</h4>
                <p style='margin: 8px 0 0 0;'>Excess Inventory Reduction</p>
                <small>Optimization algorithms</small>
            </div>
            <div class='achievement-card'>
                <h4 style='color: {ACCENT}; margin: 0;'>20%</h4>
                <p style='margin: 8px 0 0 0;'>Stockout Reduction</p>
                <small>Predictive tracking systems</small>
            </div>
            <div class='achievement-card'>
                <h4 style='color: {ACCENT}; margin: 0;'>15%</h4>
                <p style='margin: 8px 0 0 0;'>Logistics Cost Reduction</p>
                <small>Route and contract optimization</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Education & Certifications
    st.markdown("## 🎓 Education & Certifications")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>📚 Education</h4>
            <div style='margin-bottom: 1.5rem;'>
                <h5 style='color: {TEXT}; margin-bottom: 4px;'>The Catholic University of Eastern Africa</h5>
                <p style='margin: 0; color: {SUBTEXT};'>Bachelor of Commerce</p>
                <p style='margin: 0; color: {SUBTEXT};'>Business & Supply Chain Focus</p>
            </div>
            <div>
                <h5 style='color: {TEXT}; margin-bottom: 4px;'>Moringa School</h5>
                <p style='margin: 0; color: {SUBTEXT};'>Data Science, Machine Learning & AI</p>
                <p style='margin: 0; color: {SUBTEXT};'>Advanced Technical Training</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>🏅 Certifications</h4>
            <div style='margin-bottom: 1rem;'>
                <h5 style='color: {TEXT}; margin-bottom: 4px;'>International Business Machines (IBM)</h5>
                <p style='margin: 0; color: {SUBTEXT};'>Data Science Professional Certificate</p>
            </div>
            <div style='margin-bottom: 1rem;'>
                <h5 style='color: {TEXT}; margin-bottom: 4px;'>SAP Certification</h5>
                <p style='margin: 0; color: {SUBTEXT};'>Materials Management (MM)</p>
            </div>
            <div>
                <h5 style='color: {TEXT}; margin-bottom: 4px;'>Supply Chain Management</h5>
                <p style='margin: 0; color: {SUBTEXT};'>Demand Planning & Forecasting</p>
            </div>
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
        <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>Supply Chain & Forecasting Responsibilities:</h4>
        <ul>
        <li>Implement demand forecasting pipelines and inventory optimization strategies</li>
        <li>Oversee finished goods inventory management and SKU-level tracking systems</li>
        <li>Develop and maintain supply chain analytics dashboards for decision support</li>
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
                'Created predictive analytics dashboards improving sales efficiency by 15%',
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
        },
        {
            'title': 'Warehouse Assistant',
            'company': 'Ankill Solutions Ltd',
            'period': '2020 - June 2022', 
            'achievements': [
                'Automated data collection and reporting processes, reducing manual errors by 20%',
                'Managed material handovers and stock reconciliations using SAP functionalities',
                'Supported implementation of inventory management best practices',
                'Contributed to process improvement initiatives enhancing overall supply chain efficiency'
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

elif "🚀 Projects" in selected_nav:
    st.markdown("## 🚀 Supply Chain Projects")
    
    projects = [
        {
            'title': 'SForecast - Supply Chain Forecasting Platform',
            'description': 'End-to-end supply chain forecasting solution integrating multiple ML models for demand prediction, inventory optimization, and logistics planning. Features automated reporting, confidence intervals, and multi-item forecasting capabilities.',
            'technologies': ['Streamlit', 'Prophet', 'ARIMA', 'SQL', 'Python', 'Machine Learning'],
            'impact': 'Reduced forecasting errors by 25% and improved inventory turnover by 18% through accurate demand predictions',
            'status': '🚀 Production',
            'category': 'Demand Planning'
        },
        {
            'title': 'Inventory Optimization System',
            'description': 'AI-powered inventory management system that calculates optimal safety stock levels, reorder points, and service level targets. Integrates with ERP systems for real-time inventory optimization.',
            'technologies': ['Python', 'Machine Learning', 'SAP Integration', 'Optimization Algorithms'],
            'impact': 'Reduced excess inventory by 35% while maintaining 98.5% service levels and decreasing stockouts by 20%',
            'status': '🚀 Production', 
            'category': 'Inventory Management'
        },
        {
            'title': 'Logistics Route Optimization',
            'description': 'Transportation and distribution optimization system that minimizes logistics costs while meeting delivery deadlines. Includes route planning, carrier selection, and cost analysis.',
            'technologies': ['Python', 'Optimization', 'GIS', 'Data Analytics'],
            'impact': 'Achieved 15% reduction in transportation costs through optimized routing and carrier negotiations',
            'status': '🔬 Advanced Prototype',
            'category': 'Logistics Planning'
        },
        {
            'title': 'Supply Chain Analytics Dashboard',
            'description': 'Comprehensive analytics platform providing real-time visibility into supply chain performance metrics, including forecast accuracy, inventory turns, and logistics costs.',
            'technologies': ['Power BI', 'SQL', 'Python', 'Data Visualization'],
            'impact': 'Improved decision-making speed by 40% through real-time KPI monitoring and automated reporting',
            'status': '🚀 Production',
            'category': 'Supply Chain Analytics'
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
    st.markdown("## 🛠️ Supply Chain & Technical Expertise")
    
    # Skills Visualization
    st.altair_chart(create_supply_chain_skill_chart(), use_container_width=True)
    
    # Skills Categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>📈 Supply Chain Management</h4>
            <ul>
            <li>Demand Forecasting & Planning</li>
            <li>Inventory Optimization</li>
            <li>Logistics & Distribution</li>
            <li>Supply Chain Analytics</li>
            <li>Process Improvement</li>
            <li>SAP MM/PP/SD Modules</li>
            <li>Supplier Relationship Management</li>
            <li>Supply Chain Strategy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>🤖 Data Science & Analytics</h4>
            <ul>
            <li>Machine Learning & AI</li>
            <li>Statistical Analysis</li>
            <li>Time Series Forecasting</li>
            <li>Predictive Modeling</li>
            <li>Data Visualization</li>
            <li>Python Programming</li>
            <li>SQL Database Management</li>
            <li>Data Wrangling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY}; margin-bottom: 1rem;'>💻 Technologies & Tools</h4>
            <ul>
            <li>Streamlit & Web Apps</li>
            <li>Power BI & Tableau</li>
            <li>Prophet & ARIMA</li>
            <li>TensorFlow & Scikit-learn</li>
            <li>Git Version Control</li>
            <li>AWS Cloud Services</li>
            <li>Docker Containers</li>
            <li>Advanced Excel</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif "📊 Forecasting Demo" in selected_nav:
    st.markdown("## 📊 SForecast - Supply Chain Forecasting Demo")
    
    # Initialize database
    init_sforecast_db()
    
    st.markdown(f"""
    <div class='neon-card'>
        <h3 style='color: {PRIMARY}; margin-bottom: 1rem;'>🔮 SForecast - AI-Powered Supply Chain Forecasting</h3>
        <div class='readable-text'>
        Experience my end-to-end supply chain forecasting solution. This interactive demo showcases demand prediction, 
        inventory optimization, and logistics planning capabilities using machine learning and time series analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo Tabs
    demo_tabs = st.tabs(["📈 Demand Forecasting", "📦 Inventory Analysis", "🚚 Logistics Planning"])
    
    with demo_tabs[0]:
        st.subheader("Demand Forecasting with Machine Learning")
        
        uploaded_file = st.file_uploader("Upload historical demand data (CSV)", 
                                       type=['csv'], 
                                       help="File should contain date and demand quantity columns")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} records")
                
                # Show data preview
                st.subheader("Data Overview")
                st.dataframe(df.head())
                
                # Column selection
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Select Date Column", df.columns)
                with col2:
                    value_col = st.selectbox("Select Demand Column", 
                                           df.select_dtypes(include=np.number).columns)
                
                # Forecast settings
                st.subheader("Forecast Configuration")
                periods = st.slider("Forecast Horizon (days)", 30, 365, 90)
                confidence = st.slider("Confidence Level", 80, 95, 90)
                
                if st.button("Generate Demand Forecast", type="primary"):
                    with st.spinner("Creating forecast..."):
                        try:
                            # Prepare data for Prophet
                            prophet_df = df[[date_col, value_col]].copy()
                            prophet_df = prophet_df.rename(columns={date_col: 'ds', value_col: 'y'})
                            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                            prophet_df = prophet_df.dropna()
                            
                            # Run forecast
                            forecast, model = run_prophet_forecast(prophet_df, periods)
                            
                            if forecast is not None:
                                # Display results
                                st.subheader("📊 Demand Forecast Results")
                                
                                # Create visualization
                                fig = go.Figure()
                                
                                # Historical data
                                fig.add_trace(go.Scatter(
                                    x=prophet_df['ds'], y=prophet_df['y'],
                                    name='Historical Demand',
                                    line=dict(color=PRIMARY, width=3),
                                    mode='lines+markers'
                                ))
                                
                                # Forecast
                                forecast_period = forecast[forecast['ds'] > prophet_df['ds'].max()]
                                fig.add_trace(go.Scatter(
                                    x=forecast_period['ds'], y=forecast_period['yhat'],
                                    name='Demand Forecast',
                                    line=dict(color=ACCENT, width=3, dash='dash')
                                ))
                                
                                # Confidence interval
                                fig.add_trace(go.Scatter(
                                    x=forecast_period['ds'], y=forecast_period['yhat_upper'],
                                    fill=None,
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False
                                ))
                                fig.add_trace(go.Scatter(
                                    x=forecast_period['ds'], y=forecast_period['yhat_lower'],
                                    fill='tonexty',
                                    mode='lines',
                                    line=dict(width=0),
                                    fillcolor=f'rgba{tuple(int(ACCENT.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                                    name=f'{confidence}% Confidence Interval'
                                ))
                                
                                fig.update_layout(
                                    title="Demand Forecast with Confidence Intervals",
                                    xaxis_title="Date",
                                    yaxis_title="Demand Quantity",
                                    hovermode='x unified',
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast insights
                                st.subheader("🔍 Forecast Insights")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                avg_demand = prophet_df['y'].mean()
                                forecast_avg = forecast_period['yhat'].mean()
                                growth = ((forecast_avg - avg_demand) / avg_demand) * 100
                                
                                with col1:
                                    st.metric("Historical Avg Demand", f"{avg_demand:.0f}")
                                with col2:
                                    st.metric("Forecasted Avg Demand", f"{forecast_avg:.0f}")
                                with col3:
                                    st.metric("Expected Growth", f"{growth:+.1f}%")
                                
                        except Exception as e:
                            st.error(f"Forecasting error: {str(e)}")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        else:
            st.info("👆 Upload a CSV file with historical demand data, or explore the sample analysis in other tabs.")
    
    with demo_tabs[1]:
        st.subheader("Inventory Optimization Analysis")
        
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY};'>Inventory Optimization Features</h4>
            <div class='readable-text'>
            This module calculates optimal inventory parameters based on demand patterns, lead times, and service level targets.
            </div>
            
            <div style='margin-top: 1.5rem;'>
                <h5 style='color: {TEXT};'>Key Calculations:</h5>
                <ul>
                <li><strong>Safety Stock:</strong> Buffer inventory to prevent stockouts</li>
                <li><strong>Reorder Points:</strong> Inventory levels triggering new orders</li>
                <li><strong>Service Level Optimization:</strong> Balancing inventory costs and service targets</li>
                <li><strong>Economic Order Quantity:</strong> Optimal order quantities minimizing total costs</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Inventory calculator
        st.subheader("Inventory Parameter Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_demand = st.number_input("Average Daily Demand", value=100)
            demand_std = st.number_input("Demand Standard Deviation", value=20)
            lead_time = st.number_input("Lead Time (days)", value=7)
        
        with col2:
            lead_time_std = st.number_input("Lead Time Variability (days)", value=2)
            service_level = st.slider("Target Service Level (%)", 90, 99, 95)
            unit_cost = st.number_input("Unit Cost ($)", value=10.0)
        
        if st.button("Calculate Inventory Parameters"):
            # Simple inventory calculations
            z_score = {90: 1.28, 95: 1.65, 99: 2.33}.get(service_level, 1.65)
            
            safety_stock = z_score * np.sqrt((lead_time * demand_std**2) + (avg_demand**2 * lead_time_std**2))
            reorder_point = (avg_demand * lead_time) + safety_stock
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Safety Stock", f"{safety_stock:.0f} units")
            with col2:
                st.metric("Reorder Point", f"{reorder_point:.0f} units")
            with col3:
                st.metric("Service Level", f"{service_level}%")
    
    with demo_tabs[2]:
        st.subheader("Logistics Planning & Optimization")
        
        st.markdown(f"""
        <div class='neon-card'>
            <h4 style='color: {PRIMARY};'>Logistics Optimization Capabilities</h4>
            <div class='readable-text'>
            Advanced logistics planning including route optimization, carrier selection, and transportation cost analysis.
            </div>
            
            <div style='margin-top: 1.5rem;'>
                <h5 style='color: {TEXT};'>Features Include:</h5>
                <ul>
                <li><strong>Route Optimization:</strong> Minimum distance routing with constraints</li>
                <li><strong>Cost Analysis:</strong> Transportation cost modeling and optimization</li>
                <li><strong>Carrier Selection:</strong> Multi-criteria carrier evaluation</li>
                <li><strong>Delivery Scheduling:</strong> Time-based delivery optimization</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
    <p style='margin-bottom: 0.5rem;'>CK • {datetime.now().year} Chris Kimau</p>
    <p style='margin: 0; font-size: 0.9em;'>Transforming supply chains through AI-driven forecasting and data-driven optimization</p>
</div>
""", unsafe_allow_html=True)