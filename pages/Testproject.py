import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import os
import re
import json


# =========================================================
# SETUP
# =========================================================

load_dotenv()

openai = OpenAI()

os.environ["CHROMA_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

chroma = chromadb.PersistentClient(path="./my_chroma_db")

embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-large"
)

collection = chroma.get_or_create_collection(
    name="pdf_knowledge_base",
    embedding_function=embedding_function,
    configuration={"hnsw": {"space": "cosine"}}
)


# =========================================================
# PAGE
# =========================================================

st.title("PDF Knowledge Base")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file and st.button("Process PDF"):

    # -----------------------------------------------------
    # EXTRACT TEXT
    # -----------------------------------------------------

    reader = PdfReader(uploaded_file)

    original_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            original_text += text + "\n"


    if not original_text.strip():
        st.error("No readable text found.")
        st.stop()


    # -----------------------------------------------------
    # FIND SENTENCES
    # -----------------------------------------------------

    matches = list(
        re.finditer(
            r'.+?(?:[.!?](?=\s|$)|$)',
            original_text,
            re.DOTALL
        )
    )

    sentences = []

    for i, match in enumerate(matches):

        sentences.append({
            "number": i,
            "start": match.start(),
            "end": match.end(),
            "text": match.group()
        })


    st.write(f"{len(sentences)} sentences found.")


    # -----------------------------------------------------
    # ASK GPT FOR SEMANTIC BOUNDARIES
    # -----------------------------------------------------

    boundaries = []

    BATCH_SIZE = 50


    with st.spinner("Creating semantic chunks..."):

        for batch_start in range(
            0,
            len(sentences),
            BATCH_SIZE
        ):

            batch = sentences[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            numbered_text = ""

            for sentence in batch:

                numbered_text += (
                    f"[{sentence['number']}] "
                    f"{sentence['text'].strip()}\n"
                )


            prompt = f"""
Choose logical chunk boundaries for these sentences.

Return ONLY the sentence numbers where chunks should end.

Rules:

- Group sentences by meaning and topic.
- Keep related sentences together.
- Aim for chunks no longer than 500 characters.
- Never split a sentence.
- Do not rewrite or return the document text.
- Every sentence must belong to a chunk.
- The final boundary must be sentence
  {batch[-1]['number']}.

Return JSON:

{{
    "boundaries": [4, 9, 14]
}}

SENTENCES:

{numbered_text}
"""


            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                response_format={
                    "type": "json_object"
                }
            )


            result = json.loads(
                response.choices[0].message.content
            )


            batch_boundaries = result.get(
                "boundaries",
                []
            )


            # Keep only valid boundaries
            for boundary in batch_boundaries:

                if (
                    isinstance(boundary, int)
                    and
                    batch_start
                    <= boundary
                    <= batch[-1]["number"]
                ):

                    boundaries.append(boundary)


            # Always include final sentence in batch
            if batch[-1]["number"] not in boundaries:

                boundaries.append(
                    batch[-1]["number"]
                )


    boundaries = sorted(set(boundaries))


    # -----------------------------------------------------
    # CREATE CHUNKS FROM ORIGINAL TEXT
    # -----------------------------------------------------

    chunks = []

    start_sentence = 0


    for boundary in boundaries:

        # Skip invalid old boundaries
        if boundary < start_sentence:
            continue


        # If GPT made a chunk too large,
        # split it safely at sentence boundaries.

        while start_sentence <= boundary:

            end_sentence = start_sentence


            for i in range(
                start_sentence,
                boundary + 1
            ):

                start_char = sentences[
                    start_sentence
                ]["start"]

                end_char = sentences[i]["end"]

                candidate = original_text[
                    start_char:end_char
                ]


                if (
                    len(candidate) <= 500
                    or i == start_sentence
                ):

                    end_sentence = i

                else:
                    break


            start_char = sentences[
                start_sentence
            ]["start"]

            end_char = sentences[
                end_sentence
            ]["end"]


            chunks.append({
                "text": original_text[
                    start_char:end_char
                ],
                "start": start_char,
                "end": end_char
            })


            start_sentence = end_sentence + 1


    # -----------------------------------------------------
    # PRESERVE TEXT BETWEEN SENTENCES
    # -----------------------------------------------------

    for i in range(len(chunks) - 1):

        gap_start = chunks[i]["end"]
        gap_end = chunks[i + 1]["start"]

        chunks[i]["text"] += original_text[
            gap_start:gap_end
        ]

        chunks[i]["end"] = gap_end


    # Preserve text before first sentence
    if chunks[0]["start"] > 0:

        chunks[0]["text"] = (
            original_text[:chunks[0]["start"]]
            + chunks[0]["text"]
        )


    # Preserve text after final sentence
    chunks[-1]["text"] += original_text[
        chunks[-1]["end"]:
    ]


    # -----------------------------------------------------
    # VERIFY NOTHING WAS LOST
    # -----------------------------------------------------

    reconstructed = "".join(
        chunk["text"]
        for chunk in chunks
    )


    if reconstructed != original_text:

        st.error(
            "Chunk verification failed."
        )

        st.stop()


    st.success(
        "No extracted text was lost or rewritten."
    )


    # -----------------------------------------------------
    # REMOVE OLD CHUNKS FOR THIS PDF
    # -----------------------------------------------------

    try:
        collection.delete(
            where={"source": uploaded_file.name}
        )
    except Exception:
        pass


    # -----------------------------------------------------
    # STORE IN CHROMA
    # -----------------------------------------------------

    collection.add(

        ids=[
            f"{uploaded_file.name}_{i}"
            for i in range(len(chunks))
        ],

        documents=[
            chunk["text"]
            for chunk in chunks
        ],

        metadatas=[
            {
                "source": uploaded_file.name,
                "chunk": i + 1
            }
            for i in range(len(chunks))
        ]
    )


    st.success(
        f"{len(chunks)} chunks stored in ChromaDB."
    )


    # -----------------------------------------------------
    # SHOW CHUNKS
    # -----------------------------------------------------

    with st.expander("View chunks"):

        for i, chunk in enumerate(chunks):

            st.write(f"**Chunk {i + 1}**")

            st.write(chunk["text"])

            st.caption(
                f"{len(chunk['text'])} characters"
            )

            st.divider()


# =========================================================
# ASK QUESTION
# =========================================================

st.divider()

st.header("Ask the Document")

question = st.text_input(
    "Enter your question"
)


if question and st.button("Ask"):

    if not uploaded_file:

        st.error(
            "Upload and process a PDF first."
        )

        st.stop()


    # -----------------------------------------------------
    # CHECK DOCUMENT EXISTS
    # -----------------------------------------------------

    stored = collection.get(
        where={"source": uploaded_file.name}
    )


    if not stored["ids"]:

        st.error(
            "Click 'Process PDF' first."
        )

        st.stop()


    # -----------------------------------------------------
    # RETRIEVE TOP 8 CHUNKS
    # -----------------------------------------------------

    results = collection.query(

        query_texts=[question],

        n_results=min(
            8,
            len(stored["ids"])
        ),

        where={
            "source": uploaded_file.name
        }
    )


    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]


    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    sources = {}

    context = ""


    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        number = metadata["chunk"]

        similarity = 1 - distance


        sources[number] = {
            "text": document,
            "similarity": similarity
        }


        context += f"""
CHUNK {number}

{document}

---
"""


    # -----------------------------------------------------
    # GPT ANSWERS
    # -----------------------------------------------------

    prompt = f"""
Answer the question using ONLY the chunks below.

Do not use outside knowledge.

If the answer is not contained in the chunks,
say that there is not enough information.

Also identify which chunks actually support
your answer.

Return JSON:

{{
    "answer": "your answer",
    "used_chunks": [1, 2]
}}

CHUNKS:

{context}

QUESTION:

{question}
"""


    with st.spinner("Generating answer..."):

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            response_format={
                "type": "json_object"
            }
        )


    result = json.loads(
        response.choices[0].message.content
    )


    # -----------------------------------------------------
    # SHOW ANSWER
    # -----------------------------------------------------

    st.subheader("Answer")

    st.write(result["answer"])


    # -----------------------------------------------------
    # SHOW SOURCES ACTUALLY USED
    # -----------------------------------------------------

    st.subheader("Sources Used")


    for number in result.get(
        "used_chunks",
        []
    ):

        if number in sources:

            source = sources[number]


            with st.container(border=True):

                st.write(
                    f"**Chunk {number}**"
                )

                st.write(
                    f"Cosine similarity: "
                    f"{source['similarity']:.3f}"
                )

                st.write(
                    source["text"]
                )