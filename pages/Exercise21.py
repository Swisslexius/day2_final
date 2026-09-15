import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from io import StringIO
from pypdf import PdfReader
import re
import os

st.title("Exercise 2.1")

# 1.Allows the user to upload a document.
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        st.write(text)
    else:
        bytes_data = uploaded_file.getvalue()
        string_data = bytes_data.decode("utf-8")
        st.write(string_data)

# 2.Allows the user to chunk the document.
def split_into_sentences(text):
    # Split on ., !, or ? followed by whitespace, but keep the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def chunk_sentences(sentences, max_sentences):
    return [
        " ".join(sentences[i:i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]

if uploaded_file is not None and text:
    st.subheader("Chunking")
    max_sentences = st.number_input(
        "Max sentences per chunk", min_value=1, max_value=50, value=5
    )

    sentences = split_into_sentences(text)
    chunks = chunk_sentences(sentences, max_sentences)

    st.write(f"Total sentences: {len(sentences)}")
    st.write(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} ({chunk.count('.') + chunk.count('!') + chunk.count('?')} sentence-end marks)"):
            st.write(chunk)

# 3.Saves each chunk of the document (to your project directory).
# Only show the save option if we have chunks to save
if uploaded_file is not None and text and chunks:

    st.subheader("Save chunks to disk")

    # Let the user choose (or confirm) a folder name to save into.
    # Using the uploaded file's name (without extension) as a sensible default.
    base_name = os.path.splitext(uploaded_file.name)[0]
    output_dir = st.text_input("Output folder", value=f"{base_name}_chunks")

    # When the button is clicked, write each chunk to its own .txt file
    if st.button("Save chunks as text files"):

        # Create the output folder if it doesn't already exist
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []  # keep track of what we wrote, to show the user

        for i, chunk in enumerate(chunks, 1):
            # Build a filename like chunk_001.txt, chunk_002.txt, etc.
            # zero-padding (03d) keeps files in correct order when listed alphabetically
            filename = f"chunk_{i:03d}.txt"
            filepath = os.path.join(output_dir, filename)

            # Write the chunk text to disk as a plain UTF-8 text file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(chunk)

            saved_paths.append(filepath)

        # Confirm to the user what happened
        st.success(f"Saved {len(saved_paths)} chunk(s) to '{output_dir}/'")
        with st.expander("Show saved file paths"):
            for path in saved_paths:
                st.write(path)

# 4.Reads the first chunk you have saved and stores it in a variable.
CHUNKS_ROOT = "."

st.subheader("Browse previously saved chunks")

# Look for folders that match the "*_chunks" naming pattern we used when saving.
# This means the dropdown works even after a full app restart, since it just
# scans the filesystem rather than relying on anything held in memory.
existing_folders = sorted([
    d for d in os.listdir(CHUNKS_ROOT)
    if os.path.isdir(d) and "chunks" in d.lower()
])

if not existing_folders:
    st.info("No previously saved chunk folders found yet.")
else:
    # Dropdown to pick which saved document's chunks to browse
    selected_folder = st.selectbox("Select a saved document", existing_folders)

    if selected_folder:
        # List all .txt files inside that folder, sorted so chunk_001, chunk_002... stay in order
        chunk_files = sorted([
            f for f in os.listdir(selected_folder)
            if f.endswith(".txt")
        ])

        st.write(f"Found {len(chunk_files)} chunk(s) in '{selected_folder}/'")

        # Second dropdown to pick a specific chunk within that document
        selected_chunk_file = st.selectbox("Select a chunk to view", chunk_files)

        if selected_chunk_file:
            # Read and display the contents of the chosen chunk file
            chunk_path = os.path.join(selected_folder, selected_chunk_file)
            with open(chunk_path, "r", encoding="utf-8") as f:
                chunk_content = f.read()

            st.text_area("Chunk content", chunk_content, height=300)