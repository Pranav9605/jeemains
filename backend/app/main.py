# backend/app/main.py
import os, glob, json, numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.schemas import QARequest, QAResponse, PDFProcessResponse
from app.rag import JEE_RAG
from app.utils import process_solution_pdf, process_image_with_mathpix

load_dotenv()

app = FastAPI()
rag_system = JEE_RAG()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set BASE_DIR to the absolute path of the backend folder.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QA_DATA_FILE = os.path.join(BASE_DIR, "data", "qa_data.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "embeddings.npy")

@app.on_event("startup")
async def load_pdfs_and_initialize():
    """
    On startup, load all PDF files from the 'data' folder (inside the backend folder),
    process them to extract QA pairs, and initialize the FAISS vector database.
    Then, save the QA data and embeddings to disk for future use.
    """
    pdf_folder = os.path.join(BASE_DIR, "data")
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    all_qa_data = []
    if not pdf_files:
        print(f"No PDF files found in folder: {pdf_folder}")
    else:
        for pdf_file in pdf_files:
            qa_data, error = process_solution_pdf(pdf_file)
            if error:
                print(f"Error processing {pdf_file}: {error}")
            else:
                all_qa_data.extend(qa_data)
        if all_qa_data:
            rag_system.initialize_from_pdf(all_qa_data)
            try:
                with open(QA_DATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_qa_data, f, ensure_ascii=False, indent=2)
                np.save(EMBEDDINGS_FILE, rag_system.embeddings)
                print(f"Successfully built and saved vector DB with {len(all_qa_data)} QA pairs from {len(pdf_files)} files.")
            except Exception as e:
                print("Error saving vector DB:", str(e))
        else:
            print("No QA pairs found in any PDF file.")


# The rest of your endpoints remain the same...
@app.post("/process-image", response_model=QAResponse)
async def process_image_question(
    file: UploadFile = File(...),
    k: int = Form(3)
):
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        question_text = process_image_with_mathpix(tmp_path)
        os.unlink(tmp_path)
        
        if not question_text:
            raise HTTPException(status_code=400, detail="No text extracted from image.")
        
        cleaned_question_text = (
            question_text.replace('\\(', '')
                        .replace('\\)', '')
                        .replace('{', '')
                        .replace('}', '')
                        .strip()
        )
        print("Cleaned question text:", cleaned_question_text)
        
        result = rag_system.get_answer(cleaned_question_text, k)
        print("RAG result:", result)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return QAResponse(**result)
    except Exception as e:
        print("Error in process_image_question:", str(e))
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    try:
        result = rag_system.get_answer(request.question, request.k)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return QAResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
