# backend/app/utils.py
import os
import re
import pdfplumber
import requests
import base64
from PIL import Image
import io

def parse_pdf_content(full_text: str):
    """
    Parses the full text of the PDF into QA pairs.
    This function tries multiple patterns to extract question-answer pairs.
    """
    qa_data = []
    # Remove header text if present (e.g., everything before "SECTION-A")
    if "SECTION-A" in full_text:
        full_text = full_text.split("SECTION-A", 1)[1]
    
    # Define a list of regex patterns to try
    patterns = [
        # Pattern 1: Looks for "number. <question> ... Ans. (<answer>)"
        r'(\d+\.\s*(.*?))\s*Ans\.\s*\((.*?)\)',
        # Pattern 2: Looks for "number. <question> ... Answer: <answer>" (no parentheses)
        r'(\d+\.\s*(.*?))\s*Answer:\s*(.*)',
        # Pattern 3: Sometimes the answer might be on a new line after the question, e.g., "Q1. ...\nAnswer: ...\n"
        r'(\d+\.\s*(.*?))\n\s*Answer:\s*(.*)'
    ]
    
    for pat in patterns:
        regex = re.compile(pat, re.DOTALL)
        matches = regex.findall(full_text)
        for match in matches:
            # Depending on the pattern, question text might be in group 2 and answer in group 3
            question_text = match[1].strip()
            answer_text = match[2].strip()
            # Optionally: clean up extra newlines, spaces, etc.
            if question_text and answer_text:
                qa_data.append({"question": question_text, "answer": answer_text})
    
    return qa_data

def process_solution_pdf(pdf_path: str):
    """
    Processes a single PDF file to extract QA pairs.
    Returns a tuple (qa_data, error) where error is None if successful.
    """
    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        qa_data = parse_pdf_content(full_text)
        if not qa_data:
            return None, "No question-answer pairs found. Check the PDF format."
        return qa_data, None
    except Exception as e:
        return None, f"Error processing PDF: {str(e)}"

def process_image_with_mathpix(image_path: str):
    """
    Uses Mathpix API to extract text from an image.
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    data = {
        "src": f"data:image/png;base64,{img_str}",
        "formats": ["text"]
    }
    headers = {
        "app_id": os.getenv("MATHPIX_APP_ID"),  # Loaded from .env
        "app_key": os.getenv("MATHPIX_APP_KEY"),  # Loaded from .env
        "Content-type": "application/json"
    }
    response = requests.post("https://api.mathpix.com/v3/text", json=data, headers=headers)
    result = response.json()
    print("Mathpix API response:", result)
    return result.get("text", "")
