import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import numpy as np
import base64
from prophet import Prophet
from datetime import datetime, timedelta
import time

# Set page configuration
st.set_page_config(page_title="BudgetBuddy", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for an enhanced UI and UX
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #f0fff4 100%);
        color: #1a252f;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #1a252f;
        color: #e0f7fa;
        padding: 20px;
        border-right: 3px solid #27ae60;
        border-radius: 0 10px 10px 0;
    }
    .sidebar .sidebar-content h3 {
        color: #27ae60;
        font-size: 24px;
        text-align: center;
        margin-bottom: 15px;
    }
    .stHeader {
        text-align: center;
        color: #1a252f;
        font-size: 40px;
        font-weight: bold;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        margin: 10px auto;
        max-width: 800px;
    }
    .stSubheader {
        color: #27ae60;
        font-size: 28px;
        font-weight: 500;
        padding: 10px 20px;
        text-transform: uppercase;
        border-bottom: 2px solid #27ae60;
        margin-bottom: 10px;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 4px solid #27ae60;
    }
    .metric-card {
        background-color: #27ae60;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #219653;
    }
    .stTextInput>input, .stSelectbox>select, .stNumberInput>input {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
    }
    .stTextInput>input:focus, .stSelectbox>select:focus, .stNumberInput>input:focus {
        border-color: #27ae60;
        outline: none;
        box-shadow: 0 0 5px rgba(39, 174, 96, 0.5);
    }
    .collapsible {
        background-color: #ffffff;
        cursor: pointer;
        padding: 10px 20px;
        border-radius: 5px;
        margin-bottom: 5px;
        transition: all 0.3s ease;
        border: 1px solid #ddd;
    }
    .collapsible:hover {
        background-color: #f5f5f5;
    }
    .content {
        padding: 0 18px;
        display: none;
        background-color: #f9f9f9;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stProgress > div > div {
        background-color: #27ae60;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.image("logo.png", caption="BudgetBuddy", use_container_width=True)
    st.markdown("<h3>BudgetBuddy</h3>", unsafe_allow_html=True)
    st.write("Empower your finances with smart tracking and planning!")
    st.markdown("**Categories:** Shopping, Dining, Entertainment, Utilities, Transport, Food, Medicine, Books, Stationary, Tuition, Hostel, Study Materials, Snacks, Office Supplies, Travel, Subscription, Training, Rent, Groceries, Childcare, Insurance, Other")
    st.markdown("---")
    st.write("© 2025 | Designed with ❤️")
    st.write("|For You. For Us, Always |")

# Hero Section
st.markdown(
    f"""
    <div style="background-image: url('hero-bg.png'); background-size: cover; background-position: center; padding: 50px 0; text-align: center; color: #1a252f; position: relative; overflow: hidden;">
        <div style="background-color: rgba(255, 255, 255, 0.7); padding: 20px; border-radius: 10px; display: inline-block;">
            <h2 class='stHeader'>Take Control of Your Finances</h2>
            <p style="font-size: 20px; color: #27ae60; font-weight: 500;">Track, Save, and Plan with BudgetBuddy</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Expense input section
st.markdown("<div class='stSubheader'>Expense Input</div>", unsafe_allow_html=True)
st.divider()

# Initialize expenses if not present
if 'expenses' not in st.session_state:
    st.session_state['expenses'] = pd.DataFrame(columns=['category', 'amount'])

# Clear previous data option
if st.button("Clear Expense Data", key="clear_expenses"):
    st.session_state['expenses'] = pd.DataFrame(columns=['category', 'amount'])
    st.success("Expense data cleared!", icon="✅")

uploaded_file = st.file_uploader("Upload CSV (category,amount)", type=["csv"], key="file_uploader", help="Upload a CSV file with 'category' and 'amount' columns")
if uploaded_file:
    with st.spinner("Processing CSV..."):
        time.sleep(1)  # Simulate processing time
        df_uploaded = pd.read_csv(uploaded_file)
        if all(col in df_uploaded.columns for col in ['category', 'amount']):
            st.session_state['expenses'] = df_uploaded.astype({'amount': float})
            st.success("CSV uploaded successfully!", icon="✅")
        else:
            st.error("CSV must have 'category' and 'amount' columns.", icon="❌")

# Manual expense input
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image("expense-icon.png", width=50)
with col2:
    category = st.selectbox("Category", ["Shopping", "Dining", "Entertainment", "Utilities", "Transport", "Food", "Medicine", "Books", "Stationary", "Tuition", "Hostel", "Study Materials", "Snacks", "Office Supplies", "Travel", "Subscription", "Training", "Rent", "Groceries", "Childcare", "Insurance", "Other"], key="cat_select")
with col3:
    amount = st.text_input("Amount (₹)", placeholder="e.g., 300", key="amt_input")
if st.button("Add Expense", key="add_expense", type="primary", help="Add a new expense to your list"):
    if amount and amount.replace('.', '', 1).isdigit():
        new_expense = pd.DataFrame({'category': [category], 'amount': [float(amount)]})
        st.session_state['expenses'] = pd.concat([st.session_state['expenses'], new_expense], ignore_index=True)
        st.success(f"Added: {category} - ₹{amount}", icon="✅")

# Form for submitting income and expense analysis
with st.form("income_form", clear_on_submit=False):
    st.markdown("<div class='stSubheader'>Expense Analysis</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Current Expenses:**")
    if not st.session_state['expenses'].empty:
        st.dataframe(st.session_state['expenses'].style.format({'amount': '₹{:.2f}'}), use_container_width=True)
    else:
        st.write("No expenses added yet.")
    total_spent = st.session_state['expenses']['amount'].sum() if not st.session_state['expenses'].empty else 0.0
    st.markdown(f"<div class='metric-card'>Total Spent: ₹{total_spent:.2f}</div>", unsafe_allow_html=True)
    
    # Expense Visualization
    if not st.session_state['expenses'].empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        expense_by_category = st.session_state['expenses'].groupby('category')['amount'].sum()
        ax.bar(expense_by_category.index, expense_by_category.values)
        ax.set_title("Expenses by Category")
        ax.set_ylabel("Amount (₹)")
        plt.xticks(rotation=45, ha='right')
        for i, v in enumerate(expense_by_category.values):
            ax.text(i, v + 10, f'₹{v:.2f}', ha='center')
        st.pyplot(fig)
    
    st.write("**Income Input:**")
    default_income = st.session_state.get('income', 0.0)
    income = st.number_input("Monthly Income (₹)", min_value=0.0, step=100.0, value=default_income, key="income_input", help="Enter your total monthly income")
    submit_button = st.form_submit_button("Submit Income", type="primary")
    if submit_button and income > 0:
        st.session_state['income'] = income
        st.success("Income submitted successfully!", icon="✅")
    st.markdown("</div>", unsafe_allow_html=True)

# Display and analyze data
if 'income' in st.session_state:
    total_spent = st.session_state['expenses']['amount'].sum() if not st.session_state['expenses'].empty else 0.0

    # Debt and EMI Tracker
    st.markdown("<div class='stSubheader'>Debt & EMI Tracker</div>", unsafe_allow_html=True)
    st.divider()
    with st.expander("Manage Debts and EMIs", expanded=True):
        if 'debts' not in st.session_state:
            st.session_state['debts'] = []
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            debt_name = st.text_input("Debt Name", placeholder="e.g., Car Loan", key="debt_name_input")
        with col2:
            principal = st.number_input("Principal Amount (₹)", min_value=0.0, step=1000.0, key="principal_input")
        with col3:
            interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, step=0.5, key="interest_rate_input")
        with col4:
            tenure_months = st.number_input("Tenure (Months)", min_value=1, step=1, key="tenure_input")
        due_date = st.date_input("Next Due Date", key="due_date_input")
        if st.button("Add Debt/EMI", key="add_debt_btn", type="primary"):
            if debt_name and principal > 0 and interest_rate >= 0 and tenure_months > 0:
                monthly_rate = interest_rate / 12 / 100
                emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
                st.session_state['debts'].append({
                    'name': debt_name,
                    'principal': principal,
                    'interest_rate': interest_rate,
                    'tenure': tenure_months,
                    'emi': emi,
                    'remaining_principal': principal,
                    'due_date': due_date,
                    'start_date': pd.to_datetime(datetime.now().date())
                })
                st.success(f"Added {debt_name} with EMI ₹{emi:.2f}!", icon="✅")

        if st.session_state['debts']:
            df_debts = pd.DataFrame(st.session_state['debts'])
            current_date = pd.to_datetime(datetime.now().date())
            df_debts['months_paid'] = ((current_date - df_debts['start_date']).dt.days // 30).clip(lower=0)
            df_debts['remaining_tenure'] = df_debts['tenure'] - df_debts['months_paid']
            df_debts['remaining_principal'] = df_debts.apply(
                lambda row: max(row['remaining_principal'] - row['emi'] * row['months_paid'], 0) if row['remaining_tenure'] > 0 else 0, axis=1
            )
            st.write("**Debt Summary:**")
            st.dataframe(df_debts[['name', 'principal', 'emi', 'remaining_principal', 'remaining_tenure', 'due_date']].style.format({
                'principal': '₹{:.2f}',
                'emi': '₹{:.2f}',
                'remaining_principal': '₹{:.2f}'
            }), use_container_width=True)

            next_due = df_debts.loc[df_debts['remaining_tenure'] > 0, 'due_date'].min()
            if next_due and next_due <= datetime.now().date() + timedelta(days=7):
                st.warning(f"Next EMI due on {next_due.strftime('%Y-%m-%d')} is approaching!", icon="⚠️")

    # Goal-based Budgeting
    st.markdown("<div class='stSubheader'>Goal-based Budgeting</div>", unsafe_allow_html=True)
    st.divider()
    with st.expander("Set Financial Goals", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            goal_name = st.text_input("Goal Name", placeholder="e.g., Bike", key="goal_name_input")
        with col2:
            goal_amount = st.number_input("Goal Amount (₹)", min_value=0.0, step=1000.0, key="goal_amount_input")
        with col3:
            goal_months = st.number_input("Months to Achieve", min_value=1, step=1, key="goal_months_input")
        
        if goal_name and goal_amount > 0 and goal_months > 0:
            st.session_state['financial_goal'] = {
                'name': goal_name,
                'amount': goal_amount,
                'months': goal_months,
                'start_date': pd.Timestamp.now().date()
            }
            st.success(f"Goal '{goal_name}' set for ₹{goal_amount} in {goal_months} months!", icon="✅")

        if 'financial_goal' in st.session_state:
            goal_data = st.session_state['financial_goal']
            total_expenses = st.session_state['expenses']['amount'].sum() if not st.session_state['expenses'].empty else 0.0
            total_emi = sum(d['emi'] for d in st.session_state.get('debts', [])) if st.session_state.get('debts') else 0.0
            available_income = max(st.session_state.get('income', 0.0) - total_expenses - total_emi, 0)

            if len(st.session_state['expenses']) > 1:
                X = st.session_state['expenses'][['category']].copy()
                X = pd.get_dummies(X, columns=['category'])
                y = st.session_state['expenses']['amount']
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
                model.fit(X, y)
                future_X = pd.get_dummies(pd.DataFrame({'category': [goal_data['name']] * goal_data['months']}), columns=['category']).reindex(columns=X.columns, fill_value=0)
                predicted_savings = model.predict(future_X).mean() * goal_data['months']
                required_savings = min(max(available_income / goal_data['months'], predicted_savings / goal_data['months']), goal_data['amount'] / goal_data['months'])
            else:
                required_savings = min(available_income / goal_data['months'], goal_data['amount'] / goal_data['months'])

            progress = min((required_savings * goal_data['months']) / goal_data['amount'], 1.0) * 100
            st.progress(int(progress))

            st.write(f"**Goal:** {goal_data['name']} (₹{goal_data['amount']:.2f} in {goal_data['months']} months)")
            st.write(f"**Required Monthly Savings (Predicted):** ₹{required_savings:.2f}")
            st.write(f"**Progress:** {progress:.1f}%")

    # Smart Budget Plan
    st.markdown("<div class='stSubheader'>Smart Budget Plan</div>", unsafe_allow_html=True)
    st.divider()
    if st.button("Generate Budget Plan", key="generate_budget_plan"):
        with st.spinner("Generating budget plan..."):
            time.sleep(2)  # Simulate processing time
            total_expenses = st.session_state['expenses']['amount'].sum() if not st.session_state['expenses'].empty else 0.0
            total_emi = sum(d['emi'] for d in st.session_state.get('debts', [])) if st.session_state.get('debts') else 0.0
            adjusted_income = max(st.session_state.get('income', 0.0) - total_emi, 0)

            if len(st.session_state['expenses']) > 0 and 'financial_goal' in st.session_state:
                # Prepare data for Prophet
                df_prophet = st.session_state['expenses'].copy()
                if not df_prophet.empty:
                    df_prophet['ds'] = pd.date_range(start='2025-01-01', periods=len(df_prophet), freq='M')
                    df_prophet = df_prophet.rename(columns={'amount': 'y'})

                    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    try:
                        model.fit(df_prophet)
                        future = model.make_future_dataframe(periods=1, freq='M')
                        forecast = model.predict(future)
                        next_month_expense = forecast['yhat'].iloc[-1] if len(forecast) > 1 else df_prophet['y'].mean()
                    except Exception:
                        next_month_expense = df_prophet['y'].mean() if not df_prophet.empty else 0.0

                    goal_savings = st.session_state['financial_goal']['amount'] / st.session_state['financial_goal']['months']
                    budget_limit = max(adjusted_income - goal_savings, 0)
                    if budget_limit <= 0:
                        st.warning("Income is insufficient to cover expenses and goal savings. Please increase income or adjust your goal.")
                        budget_limit = adjusted_income  # Use full income as fallback

                    total_spent = st.session_state['expenses']['amount'].sum()
                    suggested_limits = {}
                    if total_spent > 0:
                        for cat in st.session_state['expenses']['category'].unique():
                            cat_total = st.session_state['expenses']['amount'][st.session_state['expenses']['category'] == cat].sum()
                            suggested_limits[cat] = budget_limit * (cat_total / total_spent)
                        suggested_limits['Predicted Next Month'] = next_month_expense * (budget_limit / total_spent) if total_spent > 0 else next_month_expense

                    with st.expander("View Budget Details", expanded=True):
                        st.write("**Suggested Monthly Budget (including predicted next month):**")
                        st.dataframe(pd.DataFrame.from_dict(suggested_limits, orient='index', columns=['Amount']).style.format({'Amount': '₹{:.2f}'}), use_container_width=True)

                    st.markdown("<div class='stSubheader'>Budget Allocation Visualization</div>", unsafe_allow_html=True)
                    st.divider()
                    fig, ax = plt.subplots(figsize=(22, 7))
                    sns.barplot(x=list(suggested_limits.keys()), y=[float(v) for v in suggested_limits.values()], palette='magma', ax=ax)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    ax.set_title("Suggested Budget Allocation", fontsize=18, pad=20, color='#1a252f')
                    for label in ax.get_xticklabels():
                        label.set_color('#34495e')
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.write("Add or upload at least one expense and set a financial goal to generate a budget plan.")