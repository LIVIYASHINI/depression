import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import re

# Set page config first
st.set_page_config(page_title="Depression Detection Dashboard", page_icon=":blue_heart:", layout="wide")
# Function to load the model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
# Load the saved depression model and vectorizer
depression_model_path = 'depression_model_rf.sav'
vectorizer_path = 'vectorizer.sav'
depression_model = load_model(depression_model_path)
vectorizer = load_model(vectorizer_path)
# Check if both were loaded successfully
if depression_model is None or vectorizer is None:
    st.error("One or more models failed to load. Please check the file paths.")
else:
    st.success("Models loaded successfully.")

# Dashboard Title and Description with Icon and Logo
st.image("logo.png", width=80)
st.title("Depression Detection Dashboard :blue_heart:")
st.markdown(
    "This dashboard predicts depression based on social media posts and visualizes trends. "
    "It includes real-time and batch prediction, word cloud generation, and temporal trend analysis."
)

# Updated keywords
depression_keywords = ["sad", "depressed", "hopeless", "down", "anxious", "crying", "empty", "worthless", "fatigue", "tired", "lost"]
non_depression_keywords = ["happy", "excited", "love", "great", "joyful", "content", "hopeful", "peaceful", "energized"]

def preprocess_text(text):
    """Preprocess text by removing URLs and unwanted characters."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    return text.lower().strip()

def enhanced_prediction(text):
    text = text.lower()
    # Check for depression or non-depression keywords
    if any(word in text for word in depression_keywords):
        return 1  # Depressed
    elif any(word in text for word in non_depression_keywords):
        return 0  # Not Depressed
    else:
        # Use sentiment analysis if no clear keyword match
        sentiment = TextBlob(text).sentiment.polarity
        # Vectorize text for model prediction
        vectorized_text = vectorizer.transform([text])
        model_prediction = depression_model.predict(vectorized_text)
        return model_prediction[0]  # Return prediction from the model

# Upload Text for Prediction
st.header("1. Predict Depression Status :speech_balloon:")
user_input = st.text_area("Enter social media text to predict:", placeholder="Type your text here...", height=150)
predict_button = st.button("Predict", use_container_width=True)

if predict_button and user_input:
    prediction = enhanced_prediction(user_input)
    if prediction == 1:
        st.markdown("### Prediction: **:red[Depressed]**")
        st.error("The text indicates signs of depression.")
    else:
        st.markdown("### Prediction: **:green[Not Depressed]**")
        st.success("The text does not indicate signs of depression.")

# Upload CSV File for Batch Prediction
st.header("2. Batch Prediction :file_folder:")
uploaded_file = st.file_uploader("Upload a CSV file with a column 'post_text'", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if 'post_text' in data.columns:
        data['post_text'] = data['post_text'].apply(preprocess_text)  # Preprocess text
        data['prediction'] = data['post_text'].apply(enhanced_prediction)
        st.write("Predictions:")
        st.write(data[['post_text', 'prediction']].replace({1: "Depressed", 0: "Not Depressed"}))
        st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv")
    else:
        st.error("CSV must have a 'post_text' column.")

# Bad words to exclude from word cloud
bad_words = ["fuck", "fucking", "shit", "bitch", "asshole", "crap"]

# Function to filter out bad words from the text
def filter_bad_words(text):
    words = text.split()
    words = [word for word in words if word.lower() not in bad_words]
    return " ".join(words)

# Visualize Word Cloud for Positive and Negative Classes
st.header("3. Word Cloud Analysis :cloud:")
if st.button("Generate Word Cloud", use_container_width=True):
    filtered_text = " ".join(data[data['prediction'] == 1]['post_text'].apply(filter_bad_words))
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="coolwarm").generate(filtered_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Monthly Trend Analysis
st.header("4. Depression Trend Analysis Over Time :chart_with_upwards_trend:")
uploaded_trend_file = st.file_uploader("Upload a CSV file with 'month_year' and 'prediction' columns for trend analysis", type="csv")

if uploaded_trend_file:
    trend_data = pd.read_csv(uploaded_trend_file, parse_dates=['month_year'])
    trend_data['month'] = trend_data['month_year'].dt.to_period('M')
    monthly_counts = trend_data.groupby(['month', 'label_text']).size().unstack().fillna(0)
    monthly_counts.columns = ['Not Depressed', 'Depressed']
    st.line_chart(monthly_counts)

# Style and Layout
st.sidebar.header("Settings :gear:")
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
       /* Set main background and text colors for dark theme */
       body {
           background-color: #121212;
           color: #E0E0E0;
       }
       .stApp {
           background-color: #121212;
           color: #E0E0E0;
       }
       
       /* Apply a dark theme to the sidebar and sidebar content */
       section[data-testid="stSidebar"] {
           background-color: #2E2E2E;
           color: #E0E0E0;
       }

       /* Style sidebar header and text */
       section[data-testid="stSidebar"] .css-vlzsci {
           color: #E0E0E0;
       }

       /* Style sidebar input boxes, buttons, and headers */
       section[data-testid="stSidebar"] .stTextInput>div>div>input,
       section[data-testid="stSidebar"] .stFileUploader>div {
           background-color: #333333;
           color: #E0E0E0;
       }
       
       section[data-testid="stSidebar"] .stButton>button {
           background-color: #6200EE;
           color: white;
           font-weight: bold;
           border-radius: 10px;
           padding: 12px;
       }
       
       section[data-testid="stSidebar"] .stButton>button:hover {
           background-color: #3700B3;
       }

       /* Set header colors */
       h1, h2, h3, h4 ,p, label {
          color: #FFFFFF;
       }
       .stButton>button {
           background-color: #6200EE;
           color: white;
           font-weight: bold;
           border-radius: 10px;
           padding: 12px;
       }
      </style>
       """, unsafe_allow_html=True
   )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: #E3F2FD;
            color: #333333;
        }
        .stApp {
            background-color: #E3F2FD;
        }
        .sidebar .sidebar-content {
            background-color: #E3F2FD;
            color: #333333;
        }
        .stTextInput>div>div>input {
            background-color: #FFFFFF;
            color: #333333;
        }
        .stButton>button {
            background-color: #6200EE;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 12px;
        }
        </style>
        """, unsafe_allow_html=True
    )

st.sidebar.write("Created by Liviya - Nov 2024")
