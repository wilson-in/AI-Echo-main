# ======================================================
# AI Echo : Your Smartest Conversational Partner
# ======================================================

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# ======================================================

st.set_page_config(
    page_title="AI Echo - Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

# ======================================================
# SAFE WORDCLOUD IMPORT
# ======================================================

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ======================================================
# PATH SETUP (PORTABLE & CORRECT)
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "model", "model.pkl")
)

DATA_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "dataset", "cleaned_senti_mapped_data.csv")
)

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================

st.sidebar.title("🧭 Navigation")

menu = st.sidebar.selectbox(
    "Go to",
    [
        "📘 Project Overview",
        "📊 EDA Dashboard",
        "🧠 Sentiment Prediction"
    ]
)

# ======================================================
# PROJECT OVERVIEW
# ======================================================

if menu == "📘 Project Overview":
    st.title("💬 AI Echo: Your Smartest Conversational Partner")

    st.markdown("""
    **AI Echo** is an end-to-end NLP project designed to analyze user reviews
    and classify sentiment into **Positive**, **Neutral**, or **Negative** categories.

    ### 🔍 Key Features
    - Text preprocessing and NLP feature engineering
    - Exploratory Data Analysis (EDA) with interactive visuals
    - Machine Learning–based sentiment classification
    - Web-based deployment using Streamlit

    ### 📌 Domain
    Customer Experience & Business Analytics
    """)

# ======================================================
# EDA DASHBOARD
# ======================================================

elif menu == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    chart = st.selectbox(
        "Select an analysis",
        [
            "Overall Sentiment Distribution",
            "Sentiment by Rating",
            "Sentiment by Platform",
            "Sentiment by Location",
            "Sentiment by ChatGPT Version",
            "Verified vs Non-Verified Users",
            "Average Review Length by Sentiment",
            "WordCloud - Positive Reviews",
            "WordCloud - Neutral Reviews",
            "WordCloud - Negative Reviews"
        ]
    )

    if chart == "Overall Sentiment Distribution":
        fig = px.bar(df["Sentiment"].value_counts(), title="Overall Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Sentiment by Rating":
        data = df.groupby("rating")["Sentiment"].value_counts().reset_index(name="count")
        fig = px.bar(data, x="rating", y="count", color="Sentiment", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Sentiment by Platform":
        data = df.groupby("platform")["Sentiment"].value_counts().reset_index(name="count")
        fig = px.bar(data, x="platform", y="count", color="Sentiment", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Sentiment by Location":
        data = df.groupby("location")["Sentiment"].value_counts().reset_index(name="count")
        fig = px.bar(data, x="location", y="count", color="Sentiment", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Sentiment by ChatGPT Version":
        data = df.groupby("version")["Sentiment"].value_counts().reset_index(name="count")
        fig = px.bar(data, x="version", y="count", color="Sentiment", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Verified vs Non-Verified Users":
        data = df.groupby("verified_purchase")["Sentiment"].value_counts().reset_index(name="count")
        fig = px.bar(data, x="verified_purchase", y="count", color="Sentiment", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Average Review Length by Sentiment":
        data = df.groupby("Sentiment")["review_length"].mean().reset_index()
        fig = px.bar(data, x="Sentiment", y="review_length", color="Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    elif "WordCloud" in chart:
        if not WORDCLOUD_AVAILABLE:
            st.warning("WordCloud library not installed.")
        else:
            sentiment = chart.split(" - ")[1].replace(" Reviews", "")
            text = " ".join(df[df["Sentiment"] == sentiment]["review"].astype(str))
            wc = WordCloud(background_color="black", colormap="viridis").generate(text)

            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

# ======================================================
# SENTIMENT PREDICTION
# ======================================================

elif menu == "🧠 Sentiment Prediction":

    st.title("🧠 Sentiment Prediction")

    user_input = st.text_area("✍️ Enter a review:")

    if st.button("🔍 Predict Sentiment"):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            confidence = model.predict_proba([user_input]).max()

            st.success(
                f"**Predicted Sentiment:** {prediction}  \n"
                f"**Confidence:** {confidence * 100:.2f}%"
            )
        else:
            st.warning("Please enter some text to analyze.")
