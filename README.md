PDF Chat Application

Description

This project is a PDF Chat Application that allows users to upload a PDF document and ask questions related to its content. The application uses advanced natural language processing models to process the PDF text, generate embeddings, and provide accurate answers to user queries.

With this application, users can interact with their PDF documents in a conversational manner, making it easier to extract relevant information quickly.

Features

Upload and process PDF documents.

Ask questions related to the content of the uploaded PDF.

Leverages Hugging Face's NLP models for question answering.

User-friendly interface built with Streamlit.

Technology Stack

Python for backend logic.

Streamlit for the web interface.

PyPDF2 for extracting text from PDF files.

Hugging Face models for text embedding and question answering.

FAISS for similarity search.


Installation
Clone the repository:
cd into your directory/ open with vscode
Create a Virtual Environment:
python -m venv env
Run the virtual environment: source env/bin/activate - for MacOS, env/Scripts/activate - for Linux, env/Scripts/activate.bat - for Windows cmd, env/Scripts/Activate.ps1 - for Windows PowerShell
Install the required dependencies:
pip install -r requirements.txt
Create OpenAI API Key and add it to your .env file (don't forget to remove "copy" extension from the .env before run)
Run the application:
streamlit run app.py
