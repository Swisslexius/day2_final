import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

st.title("Exercise 2.2")

# 1.Allows the user to copy and paste two different texts.
if "text_1" not in st.session_state:
    st.session_state.text_1 = ""

if "text_2" not in st.session_state:
    st.session_state.text_2 = ""

if "embedding_1" not in st.session_state:
    st.session_state.embedding_1 = None

if "embedding_2" not in st.session_state:
    st.session_state.embedding_2 = None

st.session_state.text_1 = st.text_area(
    label="Text 1"
)
#st.write(st.session_state.text_1)

st.session_state.text_2 = st.text_area(
    label="Text 2"
)
#st.write(st.session_state.text_2)

# 2.Creates an embedding for each chunk of text. For this you will need to use the embedding function from the OpenAI API.
client = OpenAI()

st.session_state.embedding_1 = client.embeddings.create(
    input=st.session_state.text_1, model="text-embedding-3-large"
)

#st.write(st.session_state.embedding_1.data[0].embedding)

st.session_state.embedding_2 = client.embeddings.create(
    input=st.session_state.text_2, model="text-embedding-3-large"
)

#st.write(st.session_state.embedding_2.data[0].embedding)

# 3.Displays the cosine similarity of the two embeddings.
embedding_1 = np.array(st.session_state.embedding_1.data[0].embedding)
embedding_2 = np.array(st.session_state.embedding_2.data[0].embedding)

cosine_similarity = np.dot(embedding_1, embedding_2) / (
    np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)
)

st.write(f"Cosine similarity: {cosine_similarity}")