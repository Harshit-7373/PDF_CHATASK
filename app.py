from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from transformers import pipeline  # Importing Hugging Face's pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import langchain
langchain.verbose = False

load_dotenv()

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

def main():
    st.markdown("""
        <style>
            body {
                background-color: white !important;
            }
            .streamlit-expanderHeader {
                font-size: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 8])  

    with col1:
        st.image("image(2).png", use_column_width=True)  

    with col2:
        st.title("Chat with my PDF")

    pdf = st.file_uploader("Upload your PDF File", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        knowledgeBase = process_text(text)

        query = st.text_input('Ask a question to the PDF...')

        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query:
            docs = knowledgeBase.similarity_search(query)

            model_name = "distilbert-base-cased-distilled-squad"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)

            qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

            context = " ".join([doc.page_content for doc in docs])

            answer = qa_pipeline(question=query, context=context)

            st.write(answer['answer'])

if __name__ == "__main__":
    main()
