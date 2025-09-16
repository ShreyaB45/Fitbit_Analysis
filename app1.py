import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="FitBit Comprehensive Dashboard",
    page_icon="🏃",
    layout="wide"
)

# Load Data
@st.cache_data

def load_data():
    return pd.read_csv("data/sampled_data.csv")

sample_df = load_data()

# Sidebar Filters
st.sidebar.title("Filters")
selected_user = st.sidebar.selectbox(
    "Select User ID",
    options=sample_df['Id'].unique() if 'Id' in sample_df.columns else ["All Users"]
)

# Dashboard Header
st.title("📊 FitBit Activity Dashboard")
st.markdown("""
Welcome to the **FitBit Analysis Dashboard**. This interactive tool provides statistical analysis, visualizations, and predictive modeling based on user activity data.
""")

# Basic Overview
st.subheader("Data Overview")
st.dataframe(sample_df.head())
st.markdown("---")

# Unit 1: Descriptive Statistics
st.header("Unit I: Descriptive Statistics & Distributions")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average Steps", f"{sample_df['TotalSteps'].mean():.0f}")
with col2:
    st.metric("Average Calories Burnt", f"{sample_df['Calories'].mean():.0f}")
with col3:
    st.metric("Weekend Entries", f"{sample_df['IsWeekend'].sum()}")

# Distribution Plots
fig1, ax1 = plt.subplots()
sns.histplot(sample_df['TotalSteps'], kde=True, color='blue', ax=ax1)
ax1.set_title("Distribution of Total Steps")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.histplot(sample_df['Calories'], kde=True, color='red', ax=ax2)
ax2.set_title("Distribution of Calories")
st.pyplot(fig2)

# Unit 2: Central Limit Theorem (CLT)
st.header("Unit II: Central Limit Theorem")
sample_means = [sample_df['TotalSteps'].sample(50).mean() for _ in range(1000)]
st.plotly_chart(px.histogram(
    sample_means,
    nbins=30,
    title="Sampling Distribution of Sample Means (n=50)"
), use_container_width=True)

# Unit 3: Hypothesis Testing
st.header("Unit III: Hypothesis Testing")
weekend = sample_df[sample_df['IsWeekend']]
weekday = sample_df[~sample_df['IsWeekend']]
t_stat, p_val = stats.ttest_ind(weekend['TotalSteps'], weekday['TotalSteps'])

st.plotly_chart(px.box(sample_df, x='IsWeekend', y='TotalSteps', title="Steps: Weekend vs Weekday"), use_container_width=True)

st.markdown(f"""
**T-Test Result:**
- t-statistic = {t_stat:.2f}
- p-value = {p_val:.4f}
- Interpretation: **{'Statistically Significant' if p_val < 0.05 else 'Not Significant'}**
""")

# Unit 4: Correlation & Regression
st.header("Unit IV: Correlation & Regression")
st.subheader("Steps vs Calories")
st.plotly_chart(px.scatter(sample_df, x='TotalSteps', y='Calories', trendline='ols'), use_container_width=True)

model = LinearRegression()
# Select only numeric columns to avoid string conversion errors
numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()

# Optionally, drop irrelevant columns like Id if present
if 'Id' in numeric_cols:
    numeric_cols.remove('Id')

X = sample_df[['TotalSteps']]  # Keep only desired features here
y = sample_df['Calories']

y = sample_df['Calories']
model.fit(X, y)

st.markdown(f"""
**Linear Regression Equation:**
Calories = {model.coef_[0]:.2f} × Steps + {model.intercept_:.2f}  
R² Score: {model.score(X, y):.2f}
""")

# Predict Calories Burnt
st.subheader("Predict Calories from Steps")
input_steps = st.number_input("Enter Total Steps:", min_value=0, value=5000)
predicted_cal = model.predict(np.array(input_steps).reshape(-1, 1))[0]
st.success(f"Predicted Calories Burnt: {predicted_cal:.2f}")

# Additional Correlation Heatmap
st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 5))
numeric_df = sample_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax3)

st.pyplot(fig3)

# Download Processed Data
st.sidebar.download_button(
    label="Download Data as CSV",
    data=sample_df.to_csv(index=False),
    file_name="fitbit_processed.csv"
)
