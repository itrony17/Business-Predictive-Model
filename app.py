import io
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from sklearn.ensemble import RandomForestRegressor
import os
import streamlit as st

# Securely extract the key from Streamlit's secret layer
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    
    # Inject it directly into the environment variable that the OpenAI library looks for
    os.environ["OPENAI_API_KEY"] = openai_key
except KeyError:
    st.error("Please add OPENAI_API_KEY to Advanced Settings on Streamlit Cloud.")



# --- PREMIUM DASHBOARD INTERFACE WRAPPER (DARK CUSTOM DESIGN SKIN) ---
st.set_page_config(page_title="AI Business Predictor Enterprise", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 26px !important; }
    .stButton>button { background-color: #21262d !important; color: #ecf2f8 !important; border: 1px solid #30363d !important; border-radius: 6px; width: 100%; font-weight: bold; }
    .stButton>button:hover { background-color: #30363d !important; border-color: #8b949e !important; }
    div.stAlert { background-color: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; }
    </style>
""", unsafe_allow_html=True)

# --- USER AUTHENTICATION SYSTEM ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_db" not in st.session_state:
    st.session_state.user_db = {"admin@platform.com": "password123"}

def auth_screen():
    st.title("🔐 AI Financial Platform Access")
    tab1, tab2 = st.tabs(["Sign In", "Create Account"])
    
    with tab1:
        login_email = st.text_input("Email Address", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In"):
            if login_email in st.session_state.user_db and st.session_state.user_db[login_email] == login_password:
                st.session_state.authenticated = True
                st.session_state.user_email = login_email
                st.rerun()
            else:
                st.error("Invalid credentials.")
                
    with tab2:
        reg_email = st.text_input("Choose Email Address", key="reg_email")
        reg_password = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register Account"):
            if reg_email in st.session_state.user_db:
                st.error("Account already exists.")
            elif "@" not in reg_email or len(reg_password) < 6:
                st.error("Provide a valid email and a 6+ character password.")
            else:
                st.session_state.user_db[reg_email] = reg_password
                st.success("Registration complete! Please switch to 'Sign In'.")

if not st.session_state.authenticated:
    auth_screen()
    st.stop()

# --- AI ENGINE CLASS ---
class BusinessPredictorAI:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def calculate_ltv_cac(self, arpu, churn_rate, marketing_spend, new_customers):
        churn_dec = churn_rate / 100 if churn_rate > 0 else 0.01
        ltv = arpu / churn_dec
        cac = marketing_spend / new_customers if new_customers > 0 else marketing_spend
        ltv_cac_ratio = ltv / cac if cac > 0 else ltv
        return {"LTV": ltv, "CAC": cac, "Ratio": round(ltv_cac_ratio, 2)}

    def train_engine(self, dataframe):
        required_cols = ["price", "marketing", "team_size", "material_cost", "net_profit"]
        if not all(col in dataframe.columns for col in required_cols):
            return False
        X = dataframe[["price", "marketing", "team_size", "material_cost"]]
        y = dataframe["net_profit"]
        self.model.fit(X, y)
        self.is_trained = True
        return True

    def train_mock_engine(self):
        np.random.seed(42)
        price = np.random.uniform(10, 500, 200)
        marketing = np.random.uniform(500, 50000, 200)
        team_size = np.random.randint(1, 50, 200)
        material_cost = price * np.random.uniform(0.1, 0.4, 200)
        revenue = (marketing * 1.6) + (price * 120)
        cogs = material_cost * 120
        opex = team_size * 4500 + marketing
        net_profit = revenue - (cogs + opex)
        df = pd.DataFrame({"price": price, "marketing": marketing, "team_size": team_size, "material_cost": material_cost, "net_profit": net_profit})
        self.train_engine(df)

    def predict_scenario(self, price, marketing, team_size, material_cost):
        features = np.array([[price, marketing, team_size, material_cost]])
        return self.model.predict(features)

# Initialize Session Engine
if "ai_platform" not in st.session_state:
    st.session_state.ai_platform = BusinessPredictorAI()
    st.session_state.ai_platform.train_mock_engine()

ai = st.session_state.ai_platform

# --- SIDEBAR CONTROLLER PANEL (WITH INTELLIGENCE GLOSSARY) ---
st.sidebar.markdown(f"👤 Account: **{st.session_state.user_email}**")

st.sidebar.markdown("---")
st.sidebar.header("📖 Financial Handbook")

st.sidebar.subheader("📊 Unit Economics (LTV & CAC)")
st.sidebar.info("""
**LTV (Customer Lifetime Value):** Expected sales revenue won per client before churn.
*   *Formula:* `ARPU / Churn Rate`

**CAC (Customer Acquisition Cost):** Total sales and ad overhead to win one new buyer.
*   *Formula:* `Marketing Spend / New Users`

**Health Scale Target Ratio:**
*   **>= 3x:** Highly Sustainable.
*   **1x - 3x:** Warning.
*   **< 1x:** Critical Loss.
""")

st.sidebar.subheader("📉 Net Profit & Loss Statement")
st.sidebar.info("""
A Profit & Loss statement lists net business income streams after expense deductions:

1.  **Gross Revenue:** Price $\\times$ Units Sold.
2.  **COGS (Cost of Goods Sold):** Unit Overheads $\\times$ Units Sold.
3.  **Gross Margin:** Revenue $-$ COGS.
4.  **OPEX:** Marketing $+$ Workforce payroll fixed items.
5.  **Net Profit / Loss:** Gross Margin $-$ OPEX.
""")

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Log Out"):
    st.session_state.authenticated = False
    st.rerun()

# --- INTERNATIONAL MULTI-CURRENCY CONVERSION SYSTEM ---
currency_selector = st.selectbox(
    "🌐 Localization Configuration (Reporting Currency Setting)",
    ["USD ($)", "EUR (€)", "GBP (£)", "INR (₹)", "BDT (৳)"]
)

rates = {"USD ($)": 1.0, "EUR (€)": 0.92, "GBP (£)": 0.78, "INR (₹)": 83.40, "BDT (৳)": 142.50}
curr_symbol = currency_selector.split()[-1].strip("()")
curr_factor = rates[currency_selector]

# --- CUSTOM TIMELINE & FISCAL CALENDAR SYSTEM ---
st.markdown("### 📅 Fiscal Context Settings")
f_col1, f_col2 = st.columns(2)
with f_col1:
    fiscal_year = st.selectbox("Select Target Fiscal Year Period", ["FY 2026-27", "FY 2027-28", "FY 2028-29"])
with f_col2:
    fiscal_start = st.selectbox(
        "Select Fiscal Year Cycle Start Month",
        ["January (Calendar Default)", "July (Standard BD/Aus Cycle)", "April (Standard Ind Cycle)"]
    )

if "July" in fiscal_start:
    months = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
elif "April" in fiscal_start:
    months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
else:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --- MAIN APP LAYOUT ---
st.divider()
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("⚙️ Model Parameters")
    price = st.number_input(f"Average Product Price ({curr_symbol})", min_value=1.0, value=float(75.0 * curr_factor)) / curr_factor
    mat_cost = st.number_input(f"Unit Cost / Material Cost ({curr_symbol})", min_value=0.0, value=float(20.0 * curr_factor)) / curr_factor
    marketing = st.number_input(f"Monthly Marketing Budget ({curr_symbol})", min_value=0, value=int(8000 * curr_factor)) / curr_factor
    team_size = st.number_input("Total Team Size (FTEs)", min_value=1, value=4)
    
    st.subheader("Growth Variables")
    arpu = st.number_input(f"Monthly Revenue Per User (ARPU) ({curr_symbol})", min_value=1.0, value=float(45.0 * curr_factor)) / curr_factor
    churn = st.slider("Monthly Customer Churn Rate (%)", 0.0, 100.0, 4.5)
    new_cust = st.number_input("New Customers Acquired / Mo", min_value=1, value=250)

with col2:
    st.header("🔮 AI Evaluation Summary")
    raw_predicted_profit = ai.predict_scenario(price, marketing, team_size, mat_cost)
    predicted_profit = float(raw_predicted_profit.item())
    
    disp_profit = predicted_profit * curr_factor
    if predicted_profit > 0:
        st.success(f"🟢 **Projected Net Profit ({fiscal_year}):** {curr_symbol}{disp_profit:,.2f} / month")
    else:
        st.error(f"🔴 **Projected Net Loss ({fiscal_year}):** {curr_symbol}{disp_profit:,.2f} / month")
        
    st.divider()
    metrics = ai.calculate_ltv_cac(arpu, churn, marketing, new_cust)
    
    st.subheader("Unit Economics & Sustainability")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("LTV Matrix Metric", f"{curr_symbol}{(metrics['LTV'] * curr_factor):,.2f}")
    m_col2.metric("CAC Matrix Metric", f"{curr_symbol}{(metrics['CAC'] * curr_factor):,.2f}")
    
    ratio = metrics["Ratio"]
    status_msg = "Healthy (>= 3x)" if ratio >= 3.0 else "Warning (< 3x)" if ratio >= 1.0 else "Critical (< 1x)"
    m_col3.metric("LTV : CAC Ratio Balance", f"{ratio}x", delta=status_msg, delta_color="normal" if ratio >= 3.0 else "inverse")

# --- CHARTS TIMELINE ---
st.divider()
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
profit_trend, cumulative_trend, cum_profit = [], [], 0

for i in range(12):
    sim_price = price * (1 + (i * 0.02))
    sim_market = marketing * (1 + (i * 0.03))
    m_profit_raw = ai.predict_scenario(sim_price, sim_market, team_size, mat_cost)
    m_profit = float(m_profit_raw.item())
    profit_trend.append(m_profit * curr_factor)
    cum_profit += (m_profit * curr_factor)
    cumulative_trend.append(cum_profit)

chart_df = pd.DataFrame({f"Monthly Profit ({curr_symbol})": profit_trend, f"Cumulative Runway ({curr_symbol})": cumulative_trend}, index=months)
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Monthly AI Net Profit Target")
    st.line_chart(chart_df[f"Monthly Profit ({curr_symbol})"], color="#2ecc71")
with chart_col2:
    st.subheader("Cumulative Capital Runway Position")
