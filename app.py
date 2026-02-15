import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import altair as alt

@st.cache_data
def load_data():
    return pd.read_csv("data/jobs.csv")

st.set_page_config(
    page_title="Job Matcher"
)

st.title("Job Matcher")
st.write("Sube tu CV y obtén un análisis automático")

uploaded_file = st.file_uploader(
    "Selecciona un archivo PDF",
    type=["pdf"] 
)

if uploaded_file is not None:
    texto_completo = ""
    st.success(f"Archivo `{uploaded_file.name}` cargado correctamente")
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

    df_sorted["match_percent"] =  (df_sorted["match_scores"] * 100).round(2)

    top5 = df_sorted.head(5)
    for idx, row in top5.iterrows():
        with st.expander(f"{row['title']} - {row['match_percent']}% match"):
            st.write(f"[Ver oferta]({row['link']})")
            st.write(row["description"])


    print(df_sorted[:5])
        
    st.subheader("Extras")
    st.dataframe(df_sorted[["title", "match_scores", "link"]])
