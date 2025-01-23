PDF Chat Application

Description

This project is a PDF Chat Application that allows users to upload a PDF document and ask questions related to its content. The application uses advanced natural language processing models to process the PDF text, generate embeddings, and provide accurate answers to user queries.

With this application, users can interact with their PDF documents in a conversational manner, making it easier to extract relevant information quickly.

(A) Features

1. Upload and process PDF documents.

2. Ask questions related to the content of the uploaded PDF.

3. Leverages Hugging Face's NLP models for question answering.

4. User-friendly interface built with Streamlit.

(B) Technology Stack

1. Python for backend logic.

2. Streamlit for the web interface.

3. PyPDF2 for extracting text from PDF files.

4. Hugging Face models for text embedding and question answering.

5. FAISS for similarity search.


(C) Installation
1. Clone the repository:
2. cd into your directory/ open with vscode
3. Create a Virtual Environment:
   python -m venv env
   Run the virtual environment: source env/bin/activate - for MacOS, env/Scripts/activate - for Linux, env/Scripts/activate.bat - for Windows cmd, env/Scripts/Activate.ps1 - for Windows PowerShell
  Install the required dependencies:
  pip install -r requirements.txt
  Create OpenAI API Key and add it to your .env file (don't forget to remove "copy" extension from the .env before run)
4. Run the application:
streamlit run app.py
