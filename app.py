import streamlit as st
import os
import uuid
import fitz  # PyMuPDF
import pdfplumber
import numpy as np

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PDF Q&A (RAG)", layout="centered")
st.title("PDF Q&A using RAG")

st.write("Upload a PDF and ask questions based on its content.")

# -----------------------------
# Session State Initialization
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# -----------------------------
# PDF Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                tables.append(
                    "\n".join(["\t".join(row) for row in table])
                )
    return tables


def create_chunks(text, chunk_size=500):
    chunks = []
    current = ""
    for line in text.split("\n"):
        current += line + " "
        if len(current) >= chunk_size:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


# -----------------------------
# Build Vector Store
# -----------------------------
if uploaded_file:
    st.success("PDF uploaded successfully.")

    session_id = str(uuid.uuid4())
    pdf_path = f"uploaded_{session_id}.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(pdf_path)
    tables = extract_tables_from_pdf(pdf_path)
    full_text = text + "\n".join(tables)

    chunks = create_chunks(full_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vectorstore = FAISS.from_texts(
        chunks, embedding=embeddings
    )

    st.success("PDF processed. Ask your questions below.")


# -----------------------------
# Load LLM (LOCAL – STABLE)
# -----------------------------
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0.3
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)


# -----------------------------
# Prompt
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a document-based assistant.

Answer the question using ONLY the information that directly answers it.
Do NOT include unrelated sections, headings, or additional projects.

Context:
{context}

Question:
{question}

Answer (focused and concise):
"""
)

qa_chain = LLMChain(llm=llm, prompt=prompt)


# -----------------------------
# Q&A Section
# -----------------------------
question = st.text_input("Ask a question about the PDF:")

if question and st.session_state.vectorstore is not None:
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])

    answer = qa_chain.run(
        {"context": context, "question": question}
    )

    st.subheader("Answer")
    st.write(answer)
