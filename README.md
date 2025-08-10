# BudgetBuddy

BudgetBuddy is a user-friendly financial management application built with Streamlit, designed to help you track expenses, manage debts, set financial goals, and create smart budget plans. With an intuitive interface and dynamic updates, it empowers you to take control of your finances effortlessly.
Table of Contents

## Features
Installation
Usage
Dependencies
File Structure
Contributing
License
Contact

## Features

### Expense Tracking: 
Add and upload expenses with categories like Shopping, Dining, and more.
### Income Management: 
Input and manage your monthly income for budgeting calculations.
### Debt & EMI Tracker: 
Record and monitor debts with EMI calculations and due date alerts.
### Goal-based Budgeting: 
Set financial goals (e.g., saving for a bike) with real-time progress tracking.
### Smart Budget Plan: 
Generate a tailored budget plan with predicted expenses and visualizations.
### Dynamic Updates: 
All calculations update instantly as you modify inputs.
### Custom UI: 
Responsive design with a modern gradient background and interactive elements.

## Installation

### Prerequisites

Python 3.7 or higher
Internet connection for installing dependencies

### Steps

Clone the repository:git clone https://github.com/your-username/BudgetBuddy.git
cd BudgetBuddy


Install the required dependencies:pip install -r requirements.txt


Ensure the following image files are in the project directory (C:\Users\tinku\OneDrive\Desktop\GitHub_Files\BudgetBuddy\):
logo.png
expense-icon.png
hero-bg.png


Run the application:streamlit run app.py

Open your browser and navigate to the provided local URL (e.g., http://localhost:8501).

### Usage

Sidebar: Explore category options and app information.
Expense Input: Manually add expenses or upload a CSV file with category and amount columns.
Expense Analysis: View total spent and a bar chart of expenses by category.
Income Input: Enter your monthly income and submit to enable further features.
Debt & EMI Tracker: Add debts with principal, interest rate, tenure, and due date to track EMIs.
Goal-based Budgeting: Set a financial goal with a name, amount, and timeline to track progress.
Smart Budget Plan: Generate a detailed budget plan with visualizations after adding expenses and a goal.

### Dependencies
Create a requirements.txt file with the following:
streamlit
pandas
matplotlib
seaborn
xgboost
numpy
prophet

Install them using pip install -r requirements.txt.
File Structure
BudgetBuddy/
├── app.py          # Main Streamlit application code
├── logo.png        # Sidebar logo image
├── expense-icon.png # Expense input icon
├── hero-bg.png     # Hero section background image
├── requirements.txt # Dependency list
└── README.md       # This file
