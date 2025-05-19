# Simple RAG Pipeline

This project allows you to upload documents, ask questions based on their content, and get grounded answers using a local LLM or the OpenAI API.

Upload .pdf or .txt files.

Break documents into small chunks (~300–500 tokens).

Store chunks as embeddings using FAISS.

Ask questions via a simple POST request.

Get answers from a local language model Or OpenAI GPT-3.5

## Setup Instructions:

1. Clone the Repository.
    ```bash
    git clone https://github.com/Kalikousik-14/simple-RAG-pipeline.git
    cd simple RAG pipeline

2. Install Required Libraries.
    ```pip install -r requirements.txt```

3. Set Your OpenAI API Key in .env file.

4. Run the Flask App.
    ```python app.py```

## Verification:

Verify the working by calling the apis thorugh postman or curl or browser:
1. `POST /upload` (form-data: file upload)
2. `POST /ask_local` (raw: {"question": "whatever query needed from txt or pdf"}) (for local llm) 
3. `POST /ask_openai` (raw: {"question": "whatever query needed from txt or pdf"}) (for openai) 

## Tools Used:

Embedding              sentence-transformers (MiniLM)
Vector DB              FAISS
Document Parsing       PyPDF2 / built-in fopen()
Local LLM              transformers + quantized Mistral
Cloud LLM (optional)   OpenAI API (gpt-3.5-turbo)
Web API                Flask
