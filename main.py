import streamlit as st
import googleapiclient.discovery
import pandas as pd
import re
import html
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from deep_translator import GoogleTranslator
from textblob import TextBlob
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.tokenize import RegexpTokenizer

# === Download stopwords jika belum ada ===
from nltk.corpus import stopwords
nltk.download('stopwords')

try:
    stop_words = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

stop_words.add('ya')

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load kamus kata tidak baku
kamus_data = pd.read_excel('data/kamuskatabaku.xlsx')
kata_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))

# Konfigurasi Streamlit
st.set_page_config(page_title="Machine Learning 2025", layout="wide")
st.title("Machine Learning 2025")

# Sidebar
st.sidebar.title("Menu")
menu_options = ["Crawl Dataset YT", "Preprocessing Data", "Process Text"]
selected_menu = st.sidebar.radio("Select a menu", menu_options)

if 'menu' not in st.session_state:
    st.session_state.menu = "Crawl Dataset YT"
else:
    st.session_state.menu = selected_menu

# === Fungsi Ekstraksi ID Video YouTube ===
def extract_video_id(url):
    pattern = r"(?<=v=)[\w-]+"
    match = re.search(pattern, url)
    return match.group(0) if match else None

# === Fungsi Mengambil Komentar dari YouTube ===
def fetch_youtube_comments(video_url):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyCBL9oDOq0YqRYqMLQJVvUDW1mZjcPD_sU"  # Ganti jika perlu

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL.")
        return None

    comments = []
    page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100, pageToken=page_token)
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'Timestamp': comment['publishedAt'],
                'Username': comment['authorDisplayName'],
                'VideoID': video_id,
                'Comment': comment['textDisplay'],
                'Date': comment.get('updatedAt', comment['publishedAt'])
            })

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return pd.DataFrame(comments)

# === Fungsi Preprocessing ===
def clean_comment(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def replace_taboo_words(text, kamus_tidak_baku):
    words = text.split()
    return ' '.join([kamus_tidak_baku.get(word, word) for word in words])

def stem_text(text):
    if isinstance(text, list):
        text = ' '.join(text)
    return stemmer.stem(text)

def convert_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def process_text(text):
    cleaned = clean_comment(text)
    normalized = replace_taboo_words(cleaned, kata_tidak_baku)
    tokens = tokenizer.tokenize(normalized)
    no_stopwords = [token for token in tokens if token not in stop_words]
    stemmed = stem_text(no_stopwords)
    translated = convert_english(stemmed)
    polarity = TextBlob(translated).sentiment.polarity
    sentiment = 'positive' if polarity > 0 else ('neutral' if polarity == 0 else 'negative')
    return sentiment, polarity

# Inisialisasi tokenizer regex
tokenizer = RegexpTokenizer(r'\w+')

# === Menu 1: Crawl YouTube Comments ===
if st.session_state.menu == "Crawl Dataset YT":
    video_url = st.text_input("Enter YouTube video URL:")
    if st.button("Fetch Comments"):
        if video_url:
            with st.spinner("Fetching comments..."):
                df = fetch_youtube_comments(video_url)
                if df is not None:
                    st.write(f"Total comments fetched: {len(df)}")
                    st.dataframe(df.head())
                    if st.button("Save to CSV"):
                        df.to_csv('data_debat/youtube_video_comments.csv', index=False)
                        st.success("File saved successfully.")
        else:
            st.error("Please enter a valid URL.")

# === Menu 2: Preprocessing Data ===
elif st.session_state.menu == "Preprocessing Data":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Wordcloud
        st.subheader("Wordcloud")
        text = ' '.join(df['Comment'].astype(str))
        wordcloud = WordCloud(width=800, height=400).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Clean
        df['Cleaning'] = df['Comment'].astype(str).apply(clean_comment)
        df['Normalisasi'] = df['Cleaning'].apply(lambda x: replace_taboo_words(x, kata_tidak_baku))
        df['Tokenization'] = df['Normalisasi'].apply(tokenizer.tokenize)
        df['Stopwords_Removal'] = df['Tokenization'].apply(lambda tokens: [t for t in tokens if t not in stop_words])
        df['Stemming'] = df['Stopwords_Removal'].apply(stem_text)
        df['Translated'] = df['Stemming'].apply(convert_english)
        df['Sentiment_polarity'] = df['Translated'].apply(lambda t: TextBlob(t).sentiment.polarity)
        df['Sentiment'] = df['Sentiment_polarity'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))

        st.write(df.head())

        # Save preprocessed
        df.to_csv('data/preprocessed_comments.csv', index=False)
        st.success("Preprocessed data saved.")

        # Train/Test
        st.subheader("Model Training and Accuracy")
        X = df['Translated']
        y = df['Sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# === Menu 3: Process Input Text ===
elif st.session_state.menu == "Process Text":
    input_text = st.text_area("Enter text to process:")
    if st.button("Process Text"):
        if input_text:
            sentiment, polarity = process_text(input_text)
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Polarity: {polarity:.2f}")
        else:
            st.error("Please enter some text.")
