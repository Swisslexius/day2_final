import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import re

# Load OpenAI API key from .env
load_dotenv()

client = OpenAI()

st.title("Ask Questions About a PDF")

# 1. Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:

    # Read the PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # -----------------------------------
    # CHUNK THE DOCUMENT
    # -----------------------------------

    # Split text into complete sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Target size = 500 characters
    chunk_size = 500

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        # Add complete sentence if it fits
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "

        else:

            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start new chunk with complete sentence
            current_chunk = sentence + " "

    # Save final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    st.success(f"Document divided into {len(chunks)} chunks.")

    # -----------------------------------
    # CREATE EMBEDDINGS FOR THE CHUNKS
    # -----------------------------------

    chunk_embeddings = []

    for chunk in chunks:

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk
        )

        embedding = response.data[0].embedding

        chunk_embeddings.append(embedding)

    # -----------------------------------
    # USER ASKS A QUESTION
    # -----------------------------------

    question = st.text_input(
        "Ask a question about the document"
    )

    if st.button("Ask"):

        if question:

            # Create embedding for question
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=question
            )

            question_embedding = response.data[0].embedding

            # -----------------------------------
            # FIND MOST SIMILAR CHUNK
            # -----------------------------------

            similarities = []

            for chunk_embedding in chunk_embeddings:

                similarity = cosine_similarity(
                    [question_embedding],
                    [chunk_embedding]
                )[0][0]

                similarities.append(similarity)

            # Find highest similarity
            best_chunk_number = similarities.index(
                max(similarities)
            )

            best_chunk = chunks[best_chunk_number]

            # -----------------------------------
            # ASK GPT-4o
            # -----------------------------------

            response = client.responses.create(
                model="gpt-4o",
                input=f"""
Use only the document information below to answer the question.

Document information:
{best_chunk}

Question:
{question}

Give a short and simple answer.
"""
            )

            answer = response.output_text

            # -----------------------------------
            # DISPLAY ANSWER
            # -----------------------------------

            st.subheader("Answer")

            st.write(answer)

            # -----------------------------------
            # DISPLAY SOURCE / CITATION
            # -----------------------------------

            st.subheader("Information Used")

            st.write(
                f"Chunk {best_chunk_number + 1}"
            )

            st.write(best_chunk)

        else:

            st.warning("Please enter a question.")