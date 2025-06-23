import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import joblib
import re, html
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
import nltk

# === Setup awal
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
stop_words.add('ya')

tokenizer = RegexpTokenizer(r'\w+')
stemmer = StemmerFactory().create_stemmer()

# === Fungsi bantu
def clean_comment(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def preprocess_text(text):
    cleaned = clean_comment(text)
    tokens = tokenizer.tokenize(cleaned)
    tokens = [t for t in tokens if t not in stop_words]
    stemmed = stemmer.stem(' '.join(tokens))
    translated = GoogleTranslator(source='auto', target='en').translate(stemmed)
    return translated

# === Load model dan vectorizer
try:
    model = joblib.load('data/sentiment_model.pkl')
    vectorizer = joblib.load('data/tfidf_vectorizer.pkl')
except Exception as e:
    model, vectorizer = None, None
    st.warning(f"⚠️ Model belum tersedia: {e}")

# === UI Streamlit
st.set_page_config(page_title="Analisis Sentimen Komentar", layout="wide")
st.title("💬 Analisis Sentimen Komentar YouTube")

# === Upload CSV
uploaded = st.file_uploader("📁 Upload file CSV berisi komentar & sentimen (opsional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    if 'Comment' in df.columns and 'Sentiment' in df.columns:
        st.subheader("📊 Data Komentar & Sentimen")
        st.dataframe(df[['Comment', 'Sentiment']].head())

        st.subheader("📈 Distribusi Sentimen")
        sent_count = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sent_count, labels=sent_count.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # === Tampilkan komentar berdasarkan sentimen
        st.subheader("🔍 Komentar Berdasarkan Sentimen")

        positive_comments = df[df['Sentiment'] == 'positive']
        neutral_comments = df[df['Sentiment'] == 'neutral']
        negative_comments = df[df['Sentiment'] == 'negative']

        with st.expander("🟢 Komentar Positif"):
            st.write(f"Total: {len(positive_comments)}")
            st.dataframe(positive_comments[['Comment', 'Sentiment']])

        with st.expander("🟡 Komentar Netral"):
            st.write(f"Total: {len(neutral_comments)}")
            st.dataframe(neutral_comments[['Comment', 'Sentiment']])

        with st.expander("🔴 Komentar Negatif"):
            st.write(f"Total: {len(negative_comments)}")
            st.dataframe(negative_comments[['Comment', 'Sentiment']])
    else:
        st.error("❌ File harus memiliki kolom 'Comment' dan 'Sentiment'.")

# === Analisis manual
st.subheader("🧪 Uji Sentimen Komentar Manual")
user_text = st.text_area("Masukkan komentar:")

if st.button("Analisis"):
    if user_text:
        translated = preprocess_text(user_text)
        polarity = TextBlob(translated).sentiment.polarity
        sent_tb = 'positive' if polarity > 0 else ('neutral' if polarity == 0 else 'negative')
        st.write(f"📊 Hasil (TextBlob): **{sent_tb}** — Polarity: {polarity:.2f}")

        if model and vectorizer:
            X_vec = vectorizer.transform([translated])
            sent_ml = model.predict(X_vec)[0]
            st.write(f"🤖 Hasil (Model ML): **{sent_ml}**")
        else:
            st.warning("❌ Model belum dimuat atau gagal dibaca.")
    else:
        st.warning("Komentar tidak boleh kosong.")
