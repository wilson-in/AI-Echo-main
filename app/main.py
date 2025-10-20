# ======================================================
# AI Echo : Your Smartest Conversational Partner
# Sentiment Analysis Streamlit App
# ======================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import joblib

# ======================================================
# 1️⃣ MODEL LOADING
# ======================================================
model = joblib.load(
    "Project_5-AI-Echo\model\model.pkl"
)

# ======================================================
# 2️⃣ LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    return pd.read_csv(
        "Project_5-AI-Echo\dataset\cleaned_senti_mapped_data.csv"
    )

review_df = load_data()

# ======================================================
# 3️⃣ STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Echo - Sentiment Analysis",
    layout="wide",
    page_icon="💬"
)

# ======================================================
# 4️⃣ SIDEBAR NAVIGATION
# ======================================================
option = st.sidebar.selectbox(
    "🧭 Navigation",
    ("📘 Project Explanation", "📈 EDA Charts", "💡 Model Prediction")
)

# ======================================================
# 5️⃣ DROPDOWN FOR CHARTS
# ======================================================
chart_options = {
    "Overall sentiment of user reviews": "chart1",
    "Sentiment variation by rating": "chart2",
    "Positive Reviews WordCloud" : "chart_3_pos",
    "Neutral Reviews WordCloud": "chart_3_neu",
    "Negative Reviews WordCloud": "chart_3_neg",
    "Verified users and sentiment": "chart4",
    "Average review length by sentiment": "chart5",
    "Sentiment by location": "chart6",
    "Sentiment across platforms": "chart7",
    "Sentiment by ChatGPT version": "chart8"
}

# ======================================================
# 6️⃣ EDA CHART FUNCTIONS
# ======================================================

def chart_1():
    st.subheader("Overall Sentiment of User Reviews")
    overall_sentiment = review_df['Sentiment'].value_counts()
    fig = px.bar(
        overall_sentiment,
        x=overall_sentiment.index,
        y=overall_sentiment.values,
        color=overall_sentiment.values
    )
    fig.update_layout(xaxis_title='', yaxis_title='', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def chart_2():
    st.subheader("Sentiment Variation by Rating")
    rating_sentiment = review_df.groupby('rating')['Sentiment'].value_counts().reset_index(name='count')
    fig = px.bar(
        rating_sentiment,
        x='rating',
        y='count',
        color='Sentiment'
    )
    fig.update_layout(xaxis_title='', yaxis_title='', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def chart_4():
    st.subheader("Verified Users and Sentiment")
    verified_sentiment = review_df.groupby('verified_purchase')['Sentiment'].value_counts().reset_index(name='count')
    fig = px.bar(
        verified_sentiment,
        x='verified_purchase',
        y='count',
        color='Sentiment',
        barmode='stack'
    )
    fig.update_layout(xaxis_title='', yaxis_title='', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def chart_5():
    st.subheader("Average Review Length by Sentiment")
    review_length = review_df.groupby('Sentiment')['review_length'].mean().reset_index()
    review_length['review_length'] = review_length['review_length'].round(0).astype(int)
    fig = px.bar(
        review_length,
        x='Sentiment',
        y='review_length',
        color='review_length'
    )
    fig.update_layout(xaxis_title='', yaxis_title='', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def chart_6():
    st.subheader("Sentiment by Location")
    location_sentiment = review_df.groupby('location')['Sentiment'].value_counts().reset_index(name='count')
    fig = px.bar(
        location_sentiment,
        x='location',
        y='count',
        color='Sentiment',
        barmode='stack'
    )
    fig.update_layout(xaxis_title='', yaxis_title='', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def chart_7():
    st.subheader("Sentiment Across Platforms")
    platform_sentiment = review_df.groupby('platform')['Sentiment'].value_counts().reset_index(name='count')
    fig = px.bar(
        platform_sentiment,
        x='platform',
        y='count',
        color='Sentiment',
        barmode='stack'
    )
    fig.update_layout(xaxis_title='', yaxis_title='', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def chart_8():
    st.subheader("Sentiment by ChatGPT Version")
    version_sentiment = review_df.groupby('version')['Sentiment'].value_counts().reset_index(name='count')
    fig = px.bar(
        version_sentiment,
        x='version',
        y='count',
        color='Sentiment',
        barmode='stack'
    )
    fig.update_layout(xaxis_title='', yaxis_title='', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# --- WordClouds for chart3 ---
def chart_3_pos():
    st.subheader("🟢 Positive Reviews WordCloud")
    senti_data = review_df[review_df['Sentiment'] == 'Positive']
    all_words = ' '.join(senti_data['review'])
    wc = WordCloud(width=500, height=400, random_state=21, max_font_size=80,
                   background_color='black', colormap='Greens').generate(all_words)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)

def chart_3_neu():
    st.subheader("🔵 Neutral Reviews WordCloud")
    senti_data = review_df[review_df['Sentiment'] == 'Neutral']
    all_words = ' '.join(senti_data['review'])
    wc = WordCloud(width=500, height=400, random_state=21, max_font_size=80,
                   background_color='black', colormap='Blues').generate(all_words)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)

def chart_3_neg():
    st.subheader("🔴 Negative Reviews WordCloud")
    senti_data = review_df[review_df['Sentiment'] == 'Negative']
    all_words = ' '.join(senti_data['review'])
    wc = WordCloud(width=500, height=400, random_state=21, max_font_size=80,
                   background_color='black', colormap='Reds').generate(all_words)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)

# ======================================================
# 7️⃣ MAIN PAGE LOGIC
# ======================================================

if option == "📘 Project Explanation":
    st.markdown(
        '<h1 style="color:#00c8ff; font-family:Arial; text-align:center;">💬 AI Echo: Your Smartest Conversational Partner</h1>',
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("📘 Project Overview")

    st.markdown("""
    **Objective:**  
    AI Echo is an intelligent system that analyzes user reviews to predict sentiment and uncover insights.  
    The project demonstrates how Natural Language Processing (NLP) and Machine Learning (ML) can transform textual feedback into actionable information.

    **Key Features:**

    **1️⃣ Sentiment Analysis:**  
    - Classifies user reviews into Positive 😄, Neutral 😐, and Negative 😞 sentiments.  
    - Provides probability scores to indicate prediction confidence.

    **2️⃣ Exploratory Data Analysis (EDA):**  
    - Visualizes overall sentiment distribution.  
    - Analyzes sentiment trends across ratings, platforms, locations, and ChatGPT versions.  
    - Identifies if verified users tend to leave more positive reviews.  
    - Examines review length patterns to see if longer reviews correlate with sentiment.

    **3️⃣ Keyword & Feedback Analysis:**  
    - Generates Word Clouds for each sentiment class to highlight most frequent words.  
    - Identifies common negative feedback themes for actionable insights.

    **4️⃣ Interactive UI with Streamlit:**  
    - Sidebar navigation for switching between “Project Explanation” and “Model Prediction.”  
    - Dropdown selection for viewing EDA charts.  
    - Input box for users to enter custom reviews and get sentiment predictions instantly.  
    - Dynamic visualization using Plotly and WordCloud integrated into the dashboard.
    """)

    st.markdown("""
    **5️⃣ Tech Stack Used:**<br>

    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="30"/> Python (pandas, matplotlib, Plotly, WordCloud)<br>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/streamlit/streamlit-original.svg" width="30"/> Streamlit for interactive web interface<br>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="30"/> scikit-learn for Machine Learning<br>
    📦 Joblib for model serialization<br>
    🧠 NLTK / NLP techniques for preprocessing and feature extraction
    """, unsafe_allow_html=True) 

    st.markdown("""
        **Impact:**  
        This project enables companies or developers to quickly understand user feedback, measure product satisfaction, and identify recurring pain points.  
        It’s designed to be visually appealing, interactive, and easily extendable for real-world applications.
        """)
    



elif option == "📈 EDA Charts":
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    selected_chart = st.selectbox("Select chart to display:", list(chart_options.keys()))

    chart_dispatcher = {
        "Overall sentiment of user reviews": chart_1,
        "Sentiment variation by rating": chart_2,
        "Positive Reviews WordCloud": chart_3_pos,
        "Neutral Reviews WordCloud": chart_3_neu,
        "Negative Reviews WordCloud": chart_3_neg,
        "Verified users and sentiment": chart_4,
        "Average review length by sentiment": chart_5,
        "Sentiment by location": chart_6,
        "Sentiment across platforms": chart_7,
        "Sentiment by ChatGPT version": chart_8,
        "Positive Reviews WordCloud": chart_3_pos,
        "Neutral Reviews WordCloud": chart_3_neu,
        "Negative Reviews WordCloud": chart_3_neg
    }

    chart_dispatcher[selected_chart]()

elif option == "💡 Model Prediction":
    st.title("🧠 Sentiment Prediction")
    st.markdown("Enter your review and let AI predict the sentiment!")

    user_input = st.text_area("✍️ Type your review below:")

    if st.button("🔍 Predict Sentiment"):
        if user_input.strip():
            pred = model.predict([user_input])[0]
            pred_proba = model.predict_proba([user_input])[0]
            emoji_map = {"Positive": "😄", "Neutral": "😐", "Negative": "😞"}
            st.markdown(f"### **Predicted Sentiment:** {pred} {emoji_map.get(pred, '')}")
            st.success(f"Confidence: {max(pred_proba)*100:.2f}%")
        else:
            st.warning("⚠️ Please enter a review text!")
