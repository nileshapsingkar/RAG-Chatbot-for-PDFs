# Retrieval-Augmented Generation (RAG) Chatbot

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** Chatbot using LangChain, Python, and Streamlit. The chatbot is capable of answering questions by leveraging an external knowledge base. The knowledge base is dynamically queried, and responses are generated using a large language model (LLM) to provide accurate and contextually relevant answers.

The application utilizes LangChain's tools for setting up the RAG pipeline, combining retrieval and generation for enhanced question answering.


## Requirements

To run this project, ensure you have the following Python libraries installed. You can install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Here are the core libraries used in this project:

- `streamlit` - Framework for creating interactive applications.
- `langchain` - Tool for building RAG pipelines and integrating LLMs.
- `langchain_openai` - LangChain wrapper for OpenAI models.
- `langchain_community` - Additional community-built LangChain modules.
- `PyPDF2` - For extracting text from PDF files.
- `pdfplumber` - Another PDF text extraction library (optional).
- `faiss-cpu` - FAISS library for storing and querying vector representations of documents.
- `openai` - For using OpenAI models.

## Setup Instructions

1. **Clone the Repository**  
Clone this repository to your local machine:
    
```bash
git clone https://github.com/your-username/RAG-Chatbot.git
cd RAG-Chatbot
```

2. **Install Dependencies**  
Install the required Python libraries:
```bash
pip install -r requirements.txt
 ```

3. **Obtain OpenAI API Key**  
You will need an OpenAI API key to interact with the OpenAI models. You can obtain the API key by signing up at [OpenAI](https://www.openai.com).

4. **Run the Application**  
Start the Streamlit app:
   ```bash
streamlit run app.py
   ```
This will launch the application in your browser, and you can interact with the chatbot.

## How It Works

### Task 1: Data Loading
The chatbot allows you to upload a PDF file. The content of the PDF is extracted using the PyPDF2 or pdfplumber library, and then split into smaller chunks for easier processing. The extracted content is stored in a FAISS vector store to enable fast retrieval during question answering.

### Task 2: Setting Up RAG with LangChain
The Retrieval-Augmented Generation (RAG) pipeline is set up using LangChain. It involves the following components:

- **Retriever**: A retriever that fetches the most relevant document chunks from the vector store (FAISS) based on the user's question.
- **LLMChain**: A large language model (LLM) from OpenAI, which generates answers based on the retrieved context.
- **PromptTemplate**: A custom template is provided to the LLM to ensure accurate and contextually relevant responses.

### Task 3: Building the Chatbot
The chatbot is built using Streamlit, which allows users to interact with the system. It provides a simple interface to upload PDFs and ask questions. It stores the conversation history and displays it for the user, along with the relevant source documents for each question.

## Features
- **PDF Upload**: Upload a PDF file to be used as the knowledge base.
- **Question Answering**: Ask questions based on the PDF's content. The bot retrieves and processes the most relevant sections from the PDF to generate the answers.
- **Answer History**: View the history of your questions and answers.
- **Source Documents**: Toggle to view the specific documents (or chunks) used to generate the answers.

## Example Use Cases
- **General Knowledge**: Users can ask factual questions based on the data in the uploaded PDF.
- **Contextual Questions**: The chatbot can answer specific questions about certain sections or entities present in the PDF.

**Example user input:**
- "What is the summary of the introduction section of the document?"
- "Can you tell me more about the author mentioned in the document?"

## User Interface
- **Enter OpenAI API Key**: The user needs to enter their OpenAI API key to interact with the model.
- **Upload PDF**: The user can upload a PDF file, and the content will be processed and stored for querying.
- **Ask Questions**: After uploading the PDF, users can ask questions about the document.
- **View Q&A History**: The user can view past questions and answers, with an option to view the source documents for each response.

## Example Interaction
1. User uploads a PDF and enters the OpenAI API key.
2. User asks a question, like: "Tell me about the main findings of the research."
3. The chatbot retrieves the relevant sections from the PDF, uses the OpenAI model to generate an answer, and displays the result.
4. Source documents can be revealed by toggling the button to view the parts of the PDF used to generate the answer.
