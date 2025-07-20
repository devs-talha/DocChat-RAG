# DocChat-RAG

A powerful document chat application using Retrieval-Augmented Generation (RAG) to answer questions based on your uploaded documents.

## Overview

DocChat-RAG is a web application that allows you to upload documents (PDFs, TXT, MD, CSV, DOCX) and ask questions about their content. The application uses a combination of vector embeddings and large language models to provide accurate, contextually relevant answers directly from your documents.

## Features

- **Multi-document Support**: Upload up to 20 files at once with a total size of up to 10MB
- **Multiple File Formats**: Supports TXT, MD, CSV, PDF, and DOCX files
- **Conversational Interface**: Chat-like interface powered by Chainlit
- **Retrieval-Augmented Generation**: Uses advanced RAG techniques to provide accurate answers based on document content
- **Fast Responses**: Streams responses in real-time as they are generated
- **Configurable Settings**: All configuration options managed through Pydantic settings

## Technical Architecture

DocChat-RAG is built on the following components:

1. **Document Processing Pipeline**:
   - Document loading and parsing for various file formats
   - Text chunking using RecursiveCharacterTextSplitter
   - Vector embedding generation using Hugging Face models

2. **RAG Implementation**:
   - FAISS vector store for efficient similarity search
   - LangChain retrieval chain for finding relevant document chunks
   - LLM integration for generating responses based on retrieved content

3. **User Interface**:
   - Chainlit web interface for document uploading and chat interaction
   - FastAPI integration for web server capabilities

## Setup Instructions

### Prerequisites

- Python 3.12.9
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd DocChat-RAG
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

#### Local Development

Run the application with:

```
python -m chainlit run cl_app.py
```

Or using FastAPI:

```
uvicorn main:app --reload
```

#### Docker

Build and run with Docker:

```
docker build -t docchat-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_api_key_here docchat-rag
```

## Configuration

All configuration is managed through Pydantic settings in `config.py`. You can customize the following:

### LLM Configuration
- `openai_api_key`: Your OpenAI API key (required)
- `openai_model`: The OpenAI model to use (default: "gpt-4.1-nano")
- `openai_temperature`: Temperature setting for response generation (default: 0.0)
- `openai_max_tokens`: Maximum tokens in generated responses (default: 1000)

### Embedding Configuration
- `embedding_model`: The embedding model for document vectorization (default: "all-MiniLM-L6-v2")

### Text Processing Configuration
- `chunk_size`: Size of document chunks (default: 500)
- `chunk_overlap`: Overlap between chunks to maintain context (default: 100)

### Retrieval Configuration
- `retriever_search_type`: Search algorithm type (default: "similarity")
- `retriever_k`: Number of chunks to retrieve per query (default: 4)
- `retrieval_qa_chat_prompt`: Prompt template for the RAG system (default: "langchain-ai/retrieval-qa-chat")

### File Upload Configuration
- `max_file_size_mb`: Maximum file size in MB (default: 10)
- `max_files`: Maximum number of files allowed (default: 20)

## Usage

1. Start the application
2. Upload one or more supported documents
3. Wait for processing to complete
4. Ask questions about the document content
5. Receive AI-generated answers based on your documents

## File Structure

- `cl_app.py`: Main Chainlit application with RAG implementation
- `main.py`: FastAPI integration for web serving
- `config.py`: Pydantic settings for application configuration
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker configuration for containerization
- `.env`: Environment variables (not committed to git)

## Dependencies

The application relies on several key libraries:
- LangChain for RAG orchestration
- Chainlit for the chat interface
- FastAPI for web serving
- Hugging Face for embeddings
- FAISS for vector similarity search
- PyPDF, python-docx, and other parsers for document processing
