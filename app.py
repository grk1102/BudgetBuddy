import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import numpy as np
import base64

# Set page configuration
st.set_page_config(page_title="BudgetBuddy", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a premium UI
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
    }
    .sidebar .sidebar-content {
        background-color: #1a252f;
        color: #e0f7fa;
        padding: 20px;
        border-right: 2px solid #2ecc71;
    }
    .sidebar .sidebar-content h3 {
        color: #2ecc71;
        font-size: 24px;
        text-align: center;
    }
    .stHeader {
        text-align: center;
        color: #1a252f;
        font-size: 40px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 20px;
    }
    .stSubheader {
        color: #2ecc71;
        font-size: 28px;
        font-weight: 500;
        padding: 10px;
        text-transform: uppercase;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-card {
        background-color: #2ecc71;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .collapsible {
        background-color: #ffffff;
        cursor: pointer;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        transition: all 0.3s ease;
    }
    .collapsible:hover {
        background-color: #f1f1f1;
    }
    .content {
        padding: 0 18px;
        display: none;
        background-color: #f9f9f9;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.image(r"logo.png", caption="BudgetBuddy", use_container_width=True)
    st.markdown("<h3>BudgetBuddy</h3>", unsafe_allow_html=True)
    st.write("Empower your finances with smart tracking and planning!")
    st.markdown("**Categories:** Shopping, Dining, Entertainment, Utilities, Transport, Food, Medicine, Books, Stationary, Tuition, Hostel, Study Materials, Snacks, Office Supplies, Travel, Subscription, Training, Rent, Groceries, Childcare, Insurance, Other")
    st.markdown("---")
    st.write("© 2025 | Designed with ❤️")
    st.write("|For You, For Always |")

# Hero Section
st.markdown(
    f"""
    <div style="background-image: url(r"hero-bg.png"); background-size: cover; background-position: center; padding: 50px 0; text-align: center; color: #1a252f;">
        <h2 style="font-size: 48px; font-weight: bold; text-shadow: 2px 2px 4px rgba(255,255,255,0.5);">Take Control of Your Finances</h2>
        <p style="font-size: 20px; color: #2ecc71;">Track, Save, and Plan with BudgetBuddy</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Expense input section
st.markdown("<div class='stSubheader'>Expense Input</div>", unsafe_allow_html=True)
st.divider()

# CSV upload or manual input
if 'expenses' not in st.session_state:
    st.session_state['expenses'] = []
uploaded_file = st.file_uploader("Upload CSV (category,amount)", type=["csv"], key="file_uploader", help="Upload a CSV file with 'category' and 'amount' columns")
if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    if all(col in df_uploaded.columns for col in ['category', 'amount']):
        st.session_state['expenses'] = df_uploaded.astype({'amount': float})
        st.success("CSV uploaded successfully!", icon="✅")
    else:
        st.error("CSV must have 'category' and 'amount' columns.", icon="❌")
else:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(r"expense-icon.png", width=50)
    with col2:
        category = st.selectbox("Category", ["Shopping", "Dining", "Entertainment", "Utilities", "Transport", "Food", "Medicine", "Books", "Stationary", "Tuition", "Hostel", "Study Materials", "Snacks", "Office Supplies", "Travel", "Subscription", "Training", "Rent", "Groceries", "Childcare", "Insurance", "Other"], key="cat_select")
    with col3:
        amount = st.text_input("Amount (₹)", placeholder="e.g., 300", key="amt_input")
    if st.button("Add Expense", key="add_expense", type="primary", help="Add a new expense to your list"):
        if amount.replace('.', '', 1).isdigit():
            st.session_state['expenses'].append((category, float(amount)))
            st.success(f"Added: {category} - ₹{amount}", icon="✅")
            amount = ""  # Clear input

# Form for submitting all expenses and income
with st.form("expense_form", clear_on_submit=True):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Current Expenses:**", st.session_state['expenses'] if len(st.session_state['expenses']) > 0 else "No expenses added yet.")
    col1, col2 = st.columns([1, 1])
    with col1:
        income = st.number_input("Monthly Income (₹)", min_value=0.0, value=3000.0, step=100.0, key="income_form", help="Enter your total monthly income")
    with col2:
        submit_button = st.form_submit_button("Submit Expenses", type="primary")
    if submit_button and len(st.session_state['expenses']) > 0:
        if isinstance(st.session_state['expenses'], list):
            df_expenses = pd.DataFrame(st.session_state['expenses'], columns=['category', 'amount']).astype({'amount': float})
        else:
            df_expenses = st.session_state['expenses'].copy()
        st.session_state['expenses'] = df_expenses
        st.session_state['income'] = income
        st.success("Data submitted successfully!", icon="✅")
    elif submit_button and len(st.session_state['expenses']) == 0:
        st.error("Please add or upload at least one expense.", icon="❌")
    st.markdown("</div>", unsafe_allow_html=True)

# Display and analyze data
if 'expenses' in st.session_state and 'income' in st.session_state:
    # Collapsible Expense Summary
    st.markdown("<div class='stSubheader'>Expense Analysis</div>", unsafe_allow_html=True)
    st.divider()
    with st.expander("View Expense Details", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Submitted Expenses:**")
            st.dataframe(st.session_state['expenses'].style.format({'amount': '₹{:.2f}'}), use_container_width=True)
        with col2:
            total_spent = st.session_state['expenses']['amount'].sum()
            st.markdown(f"<div class='metric-card'><h4>Total Spent</h4><h2>₹{total_spent:.2f}</h2></div>", unsafe_allow_html=True)

    # Visualization
    st.markdown("<div class='stSubheader'>Expense Visualization</div>", unsafe_allow_html=True)
    st.divider()
    fig, ax = plt.subplots(figsize=(22, 7))
    sns.barplot(x='category', y='amount', data=st.session_state['expenses'], estimator=sum, errorbar=None, palette='viridis', ax=ax)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax.set_title("Expense Breakdown", fontsize=18, pad=20, color='#1a252f')
    for label in ax.get_xticklabels():
        label.set_color('#34495e')
    plt.tight_layout()
    st.pyplot(fig)

    # Savings Goal
    st.markdown("<div class='stSubheader'>Savings Goal</div>", unsafe_allow_html=True)
    st.divider()
    with st.expander("Set & Track Savings", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(r"savings-icon.png", width=50)
            savings_goal = st.number_input("Savings Target (₹)", min_value=0.0, value=500.0, step=50.0, key="savings_goal_input", help="Set your monthly savings target")
        with col2:
            if st.button("Set Goal", key="set_goal", type="primary"):
                st.session_state['savings_goal_set'] = savings_goal
                st.success(f"Savings goal set to ₹{savings_goal:.2f}!", icon="✅")
                remaining = st.session_state['income'] - total_spent - savings_goal
                st.session_state['remaining'] = remaining
        if 'savings_goal_set' in st.session_state:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"<div class='metric-card'><h4>Savings Goal</h4><h2>₹{st.session_state['savings_goal_set']:.2f}</h2></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><h4>Remaining</h4><h2>₹{st.session_state['remaining']:.2f}</h2></div>", unsafe_allow_html=True)

    # Hybrid ML-based Budget Plan
    st.markdown("<div class='stSubheader'>Smart Budget Plan</div>", unsafe_allow_html=True)
    st.divider()
    if len(st.session_state['expenses']) > 1 and 'savings_goal_set' in st.session_state:
        # Prepare data
        X = np.array(range(len(st.session_state['expenses']))).reshape(-1, 1)
        y = st.session_state['expenses']['amount'].values

        # Linear Regression Model
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        lr_prediction = lr_model.predict([[len(st.session_state['expenses'])]])[0]

        # XGBoost Model
        xgb_model = XGBRegressor()
        categories = st.session_state['expenses']['category'].astype('category').cat.codes
        X_xgb = np.column_stack((X, categories))
        xgb_model.fit(X_xgb, y)
        xgb_prediction = xgb_model.predict(np.column_stack(([len(st.session_state['expenses'])], [categories.iloc[-1]])))

        # Hybrid Prediction (average of both models)
        hybrid_prediction = (lr_prediction + xgb_prediction) / 2
        budget_limit = st.session_state['income'] - st.session_state['savings_goal_set']
        suggested_limits = {
            cat: budget_limit * (st.session_state['expenses']['amount'][st.session_state['expenses']['category'] == cat].sum() / total_spent if total_spent > 0 else 0)
            for cat in st.session_state['expenses']['category'].unique()
        }

        # Adjust suggested limits, capping total at budget_limit
        total_suggested = sum(suggested_limits.values()) + hybrid_prediction
        if total_suggested > 0:
            factor = budget_limit / total_suggested
            suggested_limits = {cat: float(amt) * factor for cat, amt in suggested_limits.items()}
            suggested_limits['Predicted Future'] = float(hybrid_prediction) * factor

        with st.expander("View Budget Details", expanded=True):
            st.write("**Suggested Monthly Budget (including predicted future spending):**")
            st.dataframe(pd.DataFrame.from_dict(suggested_limits, orient='index', columns=['Amount']).style.format({'Amount': '₹{:.2f}'}), use_container_width=True)

        # Visualize Budget Plan
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
        st.write("Add or upload more than one expense and set a savings goal for a personalized budget plan.")