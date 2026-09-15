import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import hashlib
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()

os.environ["CHROMA_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

# Folder containing the chunks
CHUNKS_FOLDER = "GUEST v CMR OF TAXATION BC200700970_chunks"


@st.cache_resource
def get_collection():

    # Create Chroma database
    chroma_client = chromadb.PersistentClient(
        path="./my_chroma_db"
    )

    # OpenAI embedding model
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-large"
    )

    # IMPORTANT:
    # Use a collection specifically for this case
    collection = chroma_client.get_or_create_collection(
        name="guest_taxation_case",
        embedding_function=openai_ef
    )

    # Find chunk files
    folder = Path(CHUNKS_FOLDER)

    if folder.exists():

        files = (
            sorted(folder.glob("*.txt"))
            + sorted(folder.glob("*.md"))
        )

        documents = []
        metadatas = []
        ids = []

        for file in files:

            text = file.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if text:

                documents.append(text)

                # Correct metadata
                metadatas.append({
                    "source": "Guest v Commissioner of Taxation",
                    "filename": file.name
                })

                # Unique ID
                ids.append(
                    hashlib.md5(
                        file.name.encode()
                    ).hexdigest()
                )

        # Store chunks in Chroma
        if documents:

            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    return collection


collection = get_collection()


# -----------------------------
# STREAMLIT
# -----------------------------

st.title("Guest v Commissioner of Taxation — Search")

query = st.text_input("Ask a question")

client = OpenAI()


if st.button("Search") and query:

    # Retrieve 5 most relevant chunks
    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    # -----------------------------
    # SHOW RETRIEVED CHUNKS
    # -----------------------------

    st.subheader("Retrieved Chunks")

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        with st.container(border=True):

            st.write(doc)

            st.caption(
                f"Source: {meta['filename']} | "
                f"Distance: {dist:.3f}"
            )

    # -----------------------------
    # CREATE CONTEXT
    # -----------------------------

    context = "\n\n---\n\n".join(
        results["documents"][0]
    )

    # -----------------------------
    # PROMPT
    # -----------------------------

    prompt = f"""
You are a legal research assistant.

Answer the question using only the document
information provided below.

You may combine information from multiple chunks.

If the answer cannot be found in the provided
information, say that there is not enough information.

Document information:

{context}

Question:

{query}

Answer:
"""

    # -----------------------------
    # GPT-4o
    # -----------------------------

    with st.spinner("Generating response..."):

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

    # -----------------------------
    # DISPLAY ANSWER
    # -----------------------------

    st.subheader("Answer")

    st.write(
        response.choices[0].message.content
    )