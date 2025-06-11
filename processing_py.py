import io
import os
import base64
import httpx
from typing import List, Optional

# --- All Indexify, LanceDB, and Model Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import lancedb
from lancedb.pydantic import LanceModel, Vector
import fitz # PyMuPDF

# --- Configuration ---
DB_PATH = "persistent_vectordb.lance"
CACHE_DIR = "pdf_cache"
OUTPUT_DIR = "output_files"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Pydantic Models for Data Structure ---
class TextEmbeddingTable(LanceModel):
    vector: Vector(384)
    text: str
    page_number: int
    source_id: str

# --- Pre-load Models for Efficiency ---
print("Loading sentence transformer model...")
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded.")

# --- Gemini API Helper ---
async def call_gemini_api(prompt: str, is_json_response: bool = False) -> dict:
    """A generic helper to call the Gemini API."""
    api_key = "" # Handled by environment
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if is_json_response:
        payload["generationConfig"] = {"responseMimeType": "application/json"}

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = response.json()
        
        if 'candidates' not in result or not result['candidates']:
             raise ValueError("Invalid response from Gemini API")
        
        return result['candidates'][0]['content']['parts'][0]['text']


# --- NEW: Gemini-Powered Features ---
async def summarize_document(text_chunks: List[str]) -> str:
    """✨ Generates a concise summary of a document using its initial text chunks."""
    print("Generating document summary with Gemini...")
    # Use the first ~4000 characters for the summary context
    context = " ".join(text_chunks)[:4000]
    prompt = f"""
    Based on the following initial text from a research paper, please provide a concise, one-paragraph summary (3-4 sentences) of the document's likely topic and purpose.

    CONTEXT:
    {context}
    """
    try:
        summary = await call_gemini_api(prompt)
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Could not generate a summary for this document."

async def suggest_follow_up_questions(query: str, answer: str) -> List[str]:
    """✨ Generates suggested follow-up questions based on the last interaction."""
    print("Generating follow-up questions with Gemini...")
    prompt = f"""
    A researcher asked the following question: "{query}"
    The AI assistant gave this answer: "{answer}"

    Based on this interaction, suggest three insightful and distinct follow-up questions the researcher could ask to dig deeper. Return the questions as a JSON array of strings.

    Example format:
    {{
        "questions": [
            "What is the main alternative to that approach?",
            "How does this concept affect system performance?",
            "Can you provide a specific example from the document?"
        ]
    }}
    """
    try:
        response_text = await call_gemini_api(prompt, is_json_response=True)
        # The API returns a string of JSON, so we parse it.
        import json
        return json.loads(response_text).get("questions", [])
    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        return []


# --- Core Processing Pipeline ---
async def run_extraction_pipeline(file_path: str, source_id: str) -> str:
    """
    The complete data extraction pipeline for a single PDF file.
    NOW RETURNS an auto-generated summary.
    """
    print(f"Starting extraction for: {source_id}")
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunk_texts = []
    all_chunks_for_db = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text:
            texts = text_splitter.split_text(text)
            all_chunk_texts.extend(texts)
            for t in texts:
                all_chunks_for_db.append({ "text": t, "page_number": page_num + 1, "source_id": source_id })

    # ✨ NEW: Generate summary after extracting text
    summary = await summarize_document(all_chunk_texts)

    embeddings = st_model.encode(all_chunk_texts)
    
    db = lancedb.connect(DB_PATH)
    tbl = db.create_table("text_embeddings", schema=TextEmbeddingTable, exist_ok=True)
    
    data_to_add = []
    for i, chunk in enumerate(all_chunks_for_db):
        data_to_add.append({ "vector": embeddings[i], **chunk })
    tbl.add(data_to_add)
    
    print(f"Successfully processed and indexed {source_id}.")
    return summary


# --- RAG and Highlighting Functions ---
async def query_and_synthesize(query: str) -> dict:
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table("text_embeddings")
    query_embedding = st_model.encode(query)
    results = tbl.search(query_embedding).limit(5).to_pydantic(TextEmbeddingTable)
    
    if not results:
        return {"aiAnswer": "I couldn't find relevant information in the documents.", "highlightedPdfPath": None, "summaryData": None, "suggestedQuestions": []}

    gemini_response = await ask_gemini_with_context(query, results)
    
    # ✨ NEW: Generate follow-up questions
    suggested_questions = await suggest_follow_up_questions(query, gemini_response)

    try:
        quote = gemini_response.split("<quote>")[1].split("</quote>")[0].strip()
        source_chunk = next((chunk for chunk in results if quote in chunk.text), None)
        
        if source_chunk:
            highlighted_pdf_path = highlight_text_in_pdf(source_chunk.source_id, quote, source_chunk.page_number)
            vision_text = quote
            
            return {
                "aiAnswer": gemini_response,
                "highlightedPdfPath": f"/files/{os.path.basename(highlighted_pdf_path)}",
                "summaryData": {"source": source_chunk.source_id, "quote": vision_text},
                "suggestedQuestions": suggested_questions
            }
    except IndexError:
        pass
        
    return {"aiAnswer": gemini_response, "highlightedPdfPath": None, "summaryData": None, "suggestedQuestions": suggested_questions}


# --- Helper functions (ask_gemini_with_context, highlight_text_in_pdf) ---
async def ask_gemini_with_context(query: str, context_chunks: List[TextEmbeddingTable]) -> str:
    context_str = "\n---\n".join([f"Source (Page {chunk.page_number}): {chunk.text}" for chunk in context_chunks])
    prompt = f"""
    You are an AI research assistant. Answer the user's question based *only* on the provided context below.
    Your answer should be concise. After the answer, identify the single best quote from the context that supports your answer and wrap it in a <quote> tag.

    CONTEXT:
    {context_str}

    USER QUESTION:
    {query}
    
    ANSWER:
    """
    return await call_gemini_api(prompt)

def highlight_text_in_pdf(source_id: str, text_to_highlight: str, page_number: int) -> str:
    # This function's logic is unchanged
    pass