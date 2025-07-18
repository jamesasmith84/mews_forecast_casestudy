
def format_thousands(x, _):
    return f'{x/1000:.0f}k'

import streamlit as st
import pandas as pd



from PIL import Image
import base64

# Load and encode the image
logo = Image.open("Mews.png")
with open("Mews.png", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

# Display it centered
st.markdown(
    f"<div style='text-align: center;'><img src='data:image/png;base64,{encoded}' width='300'/></div>",
    unsafe_allow_html=True
)


import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Mews Forecasting Prototype", layout="wide")
st.title("📊 Mews Forecasting Prototype")

@st.cache_data
def load_data(sheet_name):
    df = pd.read_excel("Mews VP RevOps Case Study 7.25.xlsx", sheet_name=sheet_name)
    df.columns = [col.strip() for col in df.columns]
    df["Forecast Category"] = pd.Categorical(df["Forecast Category"],
        categories=["Pipeline", "Best Case", "Probable", "Commit"], ordered=True)
    return df


model_label = st.radio("Select Forecast Model", options=[
    "Model A - Balanced",
    "Model B - Probability/FC Category Weighted"
])
sheet_map = {
    "Model A - Balanced": "Forecast Model A - Data",
    "Model B - Probability/FC Category Weighted": "Forecast Model B - Data"
}
df = load_data(sheet_name=sheet_map[model_label])


symbol = st.radio("Select Currency", options=["€", "$"])
conversion_rate = 1.0 if symbol == "€" else 1.1

st.sidebar.header("🔍 Filters")
segments = df['Segment'].dropna().unique().tolist()
regions = df['Region'].dropna().unique().tolist()
strengths = df['Model Deal Strength'].dropna().unique().tolist()

selected_segment = st.sidebar.selectbox("Segment", ["All"] + segments)
selected_region = st.sidebar.selectbox("Region", ["All"] + regions)
selected_category = st.sidebar.selectbox("Forecast Category", ["All", "Pipeline", "Best Case", "Probable", "Commit"])
selected_strength = st.sidebar.selectbox("Model Deal Strength", ["All"] + strengths)
confidence_threshold = st.sidebar.slider("Minimum Confidence (Model Weighting)", 0.0, 1.0, 0.0, 0.1)

filtered = df.copy()
if selected_segment != "All":
    filtered = filtered[filtered['Segment'] == selected_segment]
if selected_region != "All":
    filtered = filtered[filtered['Region'] == selected_region]
if selected_category != "All":
    filtered = filtered[filtered['Forecast Category'] == selected_category]
if selected_strength != "All":
    filtered = filtered[filtered['Model Deal Strength'] == selected_strength]
filtered = filtered[filtered['Model Weighting'] >= confidence_threshold]

filtered["Potential Amount"] *= conversion_rate
filtered["Model Amount"] *= conversion_rate

st.subheader("📈 Summary Metrics")
col1, col2 = st.columns(2)
total_potential = filtered["Potential Amount"].sum()
total_model = filtered["Model Amount"].sum()
col1.metric("Rep Forecast (Total Pipeline)", f"{symbol}{total_potential:,.0f}")
col2.metric(f"{model_label}", f"{symbol}{total_model:,.0f}")

st.subheader("🧠 AI Forecast Commentary")
strong_deals = filtered[filtered["Model Deal Strength"].str.strip() == "Strong Deal"]
push_avg = filtered['Push Count Weighting'].mean()
confidence_avg = filtered['Model Weighting'].mean()

commentary = []
if total_model < total_potential * 0.7:
    commentary.append("🔻 Model is more conservative than rep-entered forecasts — possible overconfidence from reps.")
if push_avg > 0.1:
    commentary.append("🔁 High average push count may indicate deal slippage.")
if len(strong_deals) < len(filtered) * 0.3:
    commentary.append("⚠️ Few strong deals — potential pipeline quality issues.")
if confidence_avg > 0.6:
    commentary.append("✅ Strong confidence overall in forecast inputs.")

for c in commentary:
    st.write(c)
if not commentary:
    st.write("📊 No major issues detected.")

st.subheader("📊 Forecast Breakdown by Category (Rep vs Model + % Line)")

cat_data = filtered.groupby("Forecast Category").agg({
    "Potential Amount": "sum",
    "Model Amount": "sum"
}).reindex(["Pipeline", "Best Case", "Probable", "Commit"])

# Compute k-values for bars
cat_data_k = (cat_data / 1000).fillna(0)

# Compute % Model vs Rep
pct = (cat_data["Model Amount"] / cat_data["Potential Amount"]).replace([float("inf"), -float("inf")], 0).fillna(0) * 100

fig1, ax1 = plt.subplots(figsize=(8,4))
width = 0.35
x = range(len(cat_data_k.index))

# Bars
bars1 = ax1.bar([i - width/2 for i in x], cat_data_k["Potential Amount"], width, label="Rep Forecast")
bars2 = ax1.bar([i + width/2 for i in x], cat_data_k["Model Amount"], width, label="Model Forecast")
ax1.set_xticks(list(x))
ax1.set_xticklabels(cat_data_k.index)
ax1.set_ylabel("Forecast (k)")
ax1.set_ylim(bottom=0)

# Line (Model vs Rep %)
ax2 = ax1.twinx()
ax2.plot(list(x), pct.values, color="black", marker="o", linestyle="-", label="% Model vs Rep")
ax2.set_ylabel("% Model / Rep")
ax2.axhline(100, color="grey", linestyle="--", linewidth=1)

# Add % labels above line points
for xi, yi in zip(x, pct.values):
    ax2.annotate(f"{yi:.0f}%", xy=(xi, yi), xytext=(0,4), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8, color="black")

# Combined legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + lines2, loc="upper left")

st.pyplot(fig1)





st.subheader("📊 Forecast by Rep (Stacked by Category)")

rep_data = filtered.groupby(["Opportunity Owner", "Forecast Category"])["Model Amount"].sum().unstack(fill_value=0)
expected_cats = ["Pipeline", "Best Case", "Probable", "Commit"]
for cat in expected_cats:
    if cat not in rep_data.columns:
        rep_data[cat] = 0
rep_data = rep_data[expected_cats]
rep_data = rep_data.loc[rep_data.sum(axis=1).sort_values(ascending=False).index]

fig2, ax2 = plt.subplots(figsize=(10,5))
bottom = None
for category in rep_data.columns:
    vals = rep_data[category]
    ax2.bar(rep_data.index, vals, bottom=bottom, label=category)
    bottom = vals if bottom is None else bottom + vals

ax2.set_ylabel("Forecast (k)")
ax2.set_title("Forecast by Rep")
plt.xticks(rotation=45)
ax2.legend(title="Category")
st.pyplot(fig2)

st.subheader("📋 Opportunity-Level Comparison")
filtered["Difference"] = filtered["Potential Amount"] - filtered["Model Amount"]

# Format Close Date

# Format Age, Push Count, and Rep Forecast Accuracy
filtered["Age (Months)"] = filtered["Age (Months)"].round(1)
filtered["Sales Rep Forecast Accuracy"] = (filtered["Sales Rep Forecast Accuracy"] * 100).round(0).astype(int).astype(str) + "%"

filtered["Probability %"] = (filtered["Probability"] * 100).round(0).astype(int).astype(str) + "%"
filtered["Close Date"] = pd.to_datetime(filtered["Close Date"]).dt.date

# Model Weighting %
filtered["Model Weighting %"] = (filtered["Model Weighting"] * 100).round(0).astype(int).astype(str) + "%"

# Format currency amounts with commas
filtered["Potential Amount"] = filtered["Potential Amount"].replace({symbol: ""}, regex=True).astype(float)
filtered["Model Amount"] = filtered["Model Amount"].replace({symbol: ""}, regex=True).astype(float)
filtered["Difference"] = filtered["Difference"].replace({symbol: ""}, regex=True).astype(float)

display_df = filtered[[
    "Opportunity Name", "Opportunity Owner", "Region", "Segment", "Forecast Category",
    "Model Deal Strength", "Probability %", "Close Date", "Model Weighting %",
    "Potential Amount", "Model Amount", "Difference"
]].copy()

# Format amounts
for col in ["Potential Amount", "Model Amount", "Difference"]:
    display_df[col] = display_df[col].map(lambda x: f"{symbol}{x:,.0f}")

# Define styling function
def highlight_strength(val):
    if val.strip() == "Strong Deal":
        return "background-color: #d4edda;"  # green
    elif val.strip() == "Moderate Deal":
        return "background-color: #fff3cd;"  # orange
    elif val.strip() == "Weak Deal":
        return "background-color: #f8d7da;"  # red
    return ""

# Apply styling
styled_df = display_df.style.applymap(highlight_strength, subset=["Model Deal Strength"])
st.dataframe(styled_df, use_container_width=True)

def to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Filtered Forecast")
    return output.getvalue()

st.download_button(
    label="📥 Download Filtered Deals as Excel",
    data=to_excel(filtered),
    file_name="filtered_forecast_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

