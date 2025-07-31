import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import numpy as np

st.title("BudgetBuddy Dashboard")

# Expense input section
st.header("Enter Your Financial Data")

# CSV upload or manual input
if 'expenses' not in st.session_state:
    st.session_state['expenses'] = []
uploaded_file = st.file_uploader("Upload CSV file (format: category,amount)", type=["csv"])
if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    if all(col in df_uploaded.columns for col in ['category', 'amount']):
        st.session_state['expenses'] = df_uploaded.astype({'amount': float})
        st.success("CSV data uploaded successfully!")
    else:
        st.error("CSV must have 'category' and 'amount' columns.")
else:
    category = st.selectbox("Select Category", ["Shopping", "Dining", "Entertainment", "Utilities", "Transport", "Food", "Medicine", "Books", "Stationary", "Tuition", "Hostel", "Study Materials", "Snacks", "Office Supplies", "Travel", "Subscription", "Training", "Rent", "Groceries", "Childcare", "Insurance", "Other"], key="cat_select")
    amount = st.text_input("Amount (₹)", placeholder="e.g., 300", key="amt_input")
    if st.button("Add Expense", key="add_expense"):
        if amount.replace('.', '', 1).isdigit():
            st.session_state['expenses'].append((category, float(amount)))
            st.success(f"Added: {category} - ₹{amount}")
            amount = ""  # Clear input

# Form for submitting all expenses and income
with st.form("expense_form"):
    st.write("Current Expenses:", st.session_state['expenses'] if len(st.session_state['expenses']) > 0 else "No expenses added yet.")
    income = st.number_input("Enter your monthly income (₹)", min_value=0.0, value=3000.0, step=100.0, key="income_form")
    submit_button = st.form_submit_button("Submit All Expenses")
    if submit_button and len(st.session_state['expenses']) > 0:
        if isinstance(st.session_state['expenses'], list):
            df_expenses = pd.DataFrame(st.session_state['expenses'], columns=['category', 'amount']).astype({'amount': float})
        else:
            df_expenses = st.session_state['expenses'].copy()
        st.session_state['expenses'] = df_expenses
        st.session_state['income'] = income
        st.success("Data submitted successfully!")
    elif submit_button and len(st.session_state['expenses']) == 0:
        st.error("Please add or upload at least one expense.")

# Display and analyze data
if 'expenses' in st.session_state and 'income' in st.session_state:
    st.write("Submitted Expenses:", st.session_state['expenses'])
    total_spent = st.session_state['expenses']['amount'].sum()
    st.write(f"Total Spent: ₹{total_spent:.2f}")

    # Visualization
    st.header("Expense Visualization")
    fig, ax = plt.subplots(figsize=(15, 6))  # Increased width
    sns.barplot(x='category', y='amount', data=st.session_state['expenses'], estimator=sum, errorbar=None, palette='viridis', ax=ax)
    plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees
    ax.set_title("Expense Breakdown")
    plt.tight_layout()  # Adjust layout to prevent clipping
    st.pyplot(fig)

    # Savings Goal
    st.header("Set Your Savings Goal")
    savings_goal = st.number_input("How much do you want to save monthly? (₹)", min_value=0.0, value=500.0, step=50.0, key="savings_goal_input")
    if st.button("Set Goal", key="set_goal"):
        st.session_state['savings_goal_set'] = savings_goal
        st.success(f"Savings goal set to ₹{savings_goal:.2f}!")
        remaining = st.session_state['income'] - total_spent - savings_goal
        st.write(f"Remaining after savings: ₹{remaining:.2f}")

    # Hybrid ML-based Budget Plan
    st.header("Smart Budget Plan")
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
            factor = budget_limit / total_suggested  # Scale to fit exactly within budget_limit
            suggested_limits = {cat: float(amt) * factor for cat, amt in suggested_limits.items()}
            suggested_limits['Predicted Future'] = float(hybrid_prediction) * factor

        st.write("Suggested Monthly Budget (including predicted future spending):")
        for cat, amt in suggested_limits.items():
            st.write(f"{cat}: ₹{float(amt):.2f}")

        # Visualize Budget Plan
        fig, ax = plt.subplots(figsize=(15, 6))  # Increased width
        sns.barplot(x=list(suggested_limits.keys()), y=[float(v) for v in suggested_limits.values()], palette='magma', ax=ax)
        plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees
        ax.set_title("Suggested Budget Allocation")
        plt.tight_layout()  # Adjust layout to prevent clipping
        st.pyplot(fig)
    else:
        st.write("Add or upload more than one expense and set a savings goal for a personalized budget plan.")