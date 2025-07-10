import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load the saved dataframe and FAISS index
@st.cache_resource
def load_resources():
    df = pd.read_pickle("legal_df.pkl")
    index = faiss.read_index("legal_index.index")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, index, model

df, index, model = load_resources()

def get_best_answer(user_input):
    query_embedding = model.encode([user_input])
    D, I = index.search(query_embedding, k=1)
    best_distance = D[0][0]
    best_index = I[0][0]
    threshold = 200.0
    if best_distance > threshold:
        return "Sorry, I don't know the answer to that."
    return df.iloc[best_index]['answer']

st.title("AI Legal Assistant Chatbot")

user_input = st.text_input("Ask your legal question:")

if user_input:
    answer = get_best_answer(user_input)
    st.write(f"**Answer:** {answer}")
