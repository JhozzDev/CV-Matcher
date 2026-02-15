import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

@st.cache_data
def load_data():
    return pd.read_csv("data/jobs.csv")

st.set_page_config(
    page_title="Job Matcher",
    layout="wide" 
)
st.title("Job Matcher")

uploaded_file = st.file_uploader(
    "Selecciona un archivo PDF",
    type=["pdf"]  # Solo permite PDFs
)


if uploaded_file is not None:
    texto_completo = ""
    with pdfplumber.open(uploaded_file) as pdf:
            texto_completo = ""
            for pagina in pdf.pages:
                texto = pagina.extract_text()
                texto_completo += texto + "\n"

if st.button("Iniciar") and uploaded_file is not None:
    
    df = load_data()
    cv_text = texto_completo

    documents = df["description"].fillna("").tolist()
    documents.insert(0, cv_text)

    vector = TfidfVectorizer()
    x = vector.fit_transform(documents)

    score = cosine_similarity(x[0:1], x[1:]).flatten()

    df["match_scores"] = score
    df_sorted = df.sort_values("match_scores", ascending=False)

    print(df_sorted[:5])
    st.dataframe(df_sorted[["title", "match_scores", "link"]])
