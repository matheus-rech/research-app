import os
import shutil
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import our custom processing logic
from processing import run_extraction_pipeline, query_and_synthesize

# Note: This application requires the GOOGLE_API_KEY environment variable to be set for full functionality.

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Research Assistant API",
    description="API for processing documents and answering questions with Gemini.",
    version="1.1.0"
)

# --- Static File Serving ---
os.makedirs("output_files", exist_ok=True)
app.mount("/files", StaticFiles(directory="output_files"), name="files")

# --- Pydantic Models for API Data Validation (Updated) ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    aiAnswer: str
    highlightedPdfPath: Optional[str]
    summaryData: Optional[Dict]
    suggestedQuestions: Optional[List[str]]  # NEW

class UploadResponse(BaseModel):
    filename: str
    message: str
    summary: Optional[str]  # NEW

# --- API Endpoints ---
@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Handles PDF file uploads. It saves the file and synchronously
    runs the processing pipeline, returning the auto-generated summary.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    file_path = os.path.join("pdf_cache", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run the pipeline and get the summary
    summary = await run_extraction_pipeline(file_path=file_path, source_id=file.filename)
    
    return {
        "filename": file.filename,
        "message": "File processed successfully.",
        "summary": summary
    }

@app.post("/api/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Handles a user's query and now returns suggested follow-up questions.
    """
    try:
        result = await query_and_synthesize(request.query)
        return result
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Research Assistant API. See /docs for documentation."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)