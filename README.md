AI Research Assistant
This project is a full-stack application that acts as an intelligent research assistant. It allows users to upload PDF documents, and then ask complex, natural-language questions about the content. The application uses a Retrieval-Augmented Generation (RAG) pipeline with the Gemini API to provide accurate, context-aware answers, create on-the-fly document summaries, and suggest insightful follow-up questions.
Features
PDF Upload & Processing: Upload local PDF files to create a persistent Knowledge Base.
Auto-Summarization: Automatically generates a concise summary for each uploaded document using Gemini.
Conversational Q&A: Ask questions in natural language and receive synthesized answers grounded in the document's content.
Source Highlighting: Automatically generates a new version of the source PDF with the relevant quote highlighted.
Suggested Questions: Provides AI-generated follow-up questions to guide research.
Exportable Summary: Compiles all findings into a summary table that can be exported as a CSV file.
Project Structure
/ai-research-assistant/
|-- backend/
|   |-- main.py             # FastAPI app, API endpoints
|   |-- processing.py       # Core data processing and Gemini logic
|   |-- requirements.txt    # Python dependencies
|   |-- pdf_cache/          # Stores uploaded PDFs
|   |-- output_files/       # Stores highlighted PDFs
|   |-- persistent_vectordb.lance/ # The vector database
|
|-- frontend/
|   |-- index.html          # The single-page frontend application
|
|-- README.md


Setup and Installation
Follow these steps to set up and run the application on your local machine.
Prerequisites
Python 3.9 or higher
An internet connection for downloading models and calling APIs
1. Set up the Backend
First, navigate to the backend directory and set up a virtual environment.
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate


Next, install all the required Python packages using the requirements.txt file.
# Install dependencies
pip install -r requirements.txt


2. Run the Application
Start the Backend Server
With your virtual environment still active, start the FastAPI server from within the backend directory.
# Run the FastAPI server
uvicorn main:app --reload


You should see output indicating the server is running, typically on http://127.0.0.1:8000.
Launch the Frontend
Navigate to the frontend directory in your file explorer and open the index.html file in your preferred web browser (like Chrome, Firefox, or Edge).
That's it! The application should now be fully functional. You can start by uploading a PDF and then asking questions about it.
