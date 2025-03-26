# frontend/app.py
import streamlit as st
import requests
import os
import io
import pandas as pd
import re
from PIL import Image

# Set the backend URL; adjust if necessary.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def process_single_image(file_obj, k=3):
    """
    Sends a single image file to the backend /process-image endpoint.
    Returns the predicted answer as provided by the backend.
    """
    files = {"file": (file_obj.name, file_obj.getvalue(), file_obj.type)}
    data = {"k": k}
    try:
        response = requests.post(f"{BACKEND_URL}/process-image", files=files, data=data)
        if response.status_code == 200:
            json_data = response.json()
            return json_data.get("predicted_answer", "No answer returned")
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Exception: {str(e)}"

def extract_option(answer_text):
    """
    Extracts a single digit (1-4) from the answer text.
    If none is found, returns the original text.
    """
    match = re.search(r'\b([1-4])\b', answer_text)
    if match:
        return match.group(1)
    return answer_text

def generate_excel(results, filename="output_answers.xlsx"):
    """
    Given a list of results (each a dict with columns), create an Excel file in memory and return bytes.
    """
    df = pd.DataFrame(results)
    # Desired output columns; customize as needed.
    desired_cols = ["Question Paper Number/Name", "Subject", "Chapters", "Theoretical",
                    "Question Number", "Correct Answers", "GPTwithrag"]
    cols = [col for col in desired_cols if col in df.columns] + [c for c in df.columns if c not in desired_cols]
    df = df[cols]
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        data = output.getvalue()
    return data

def main():
    st.set_page_config(page_title="JEE RAG Expert", layout="wide")
    st.title("JEE RAG Expert")
    st.info("The backend has preloaded the QA pairs from PDFs.")

    # --- Section 1: Single Image Query ---
    st.header("Single Image Question")
    single_img = st.file_uploader("Upload a Question Image", type=["png", "jpg", "jpeg"], key="single_img")
    k_value = st.slider("Number of References", 1, 5, 3, key="k_single")
    if single_img:
        if st.button("Get Answer for Image"):
            with st.spinner("Processing image..."):
                try:
                    image = Image.open(io.BytesIO(single_img.getvalue()))
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    answer_text = process_single_image(single_img, k=k_value)
                    option = extract_option(answer_text)
                    st.success("Answer generated!")
                    st.subheader("Predicted Answer Option:")
                    st.write(option)
                except Exception as e:
                    st.error(f"Failed to process image question: {str(e)}")
    
    st.markdown("---")
    
    # --- Section 2: Bulk Processing and Excel Output ---
    st.header("Bulk Process Multiple Images and Export to Excel")
    bulk_imgs = st.file_uploader("Upload Multiple Question Images", type=["png", "jpg", "jpeg"],
                                   key="bulk_imgs", accept_multiple_files=True)
    k_bulk = st.slider("Number of References for Bulk", 1, 5, 3, key="k_bulk")
    if bulk_imgs:
        if st.button("Generate Excel with Answers"):
            results = []
            for idx, img_file in enumerate(bulk_imgs, start=1):
                st.write(f"Processing image {idx} of {len(bulk_imgs)}: {img_file.name}")
                answer_text = process_single_image(img_file, k=k_bulk)
                # Extract just the answer option (e.g., "1", "2", "3", or "4")
                answer_option = extract_option(answer_text)
                results.append({
                    "Question Paper Number/Name": "Exam paper 1",  # Customize as needed
                    "Subject": "Physics",                           # Customize as needed
                    "Chapters": "algebra",                          # Customize as needed
                    "Theoretical": "N",                             # Customize as needed
                    "Question Number": idx,
                    "Correct Answers": "",                          # Optionally fill if available
                    "GPTwithrag": answer_option
                })
            # Generate Excel file as bytes
            excel_bytes = generate_excel(results)
            st.download_button(label="Download Answers Excel",
                               data=excel_bytes,
                               file_name="output_answers.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    st.markdown("---")
    
    # --- Section 3: Text Question Query ---
    st.header("Ask a Text Question")
    text_question = st.text_area("Enter your question:", height=150, placeholder="Type your physics/math question here...", key="text_q")
    k_text = st.slider("Number of References (Text Question)", 1, 5, 3, key="k_text")
    if st.button("Get Answer (Text)"):
        if text_question:
            with st.spinner("Processing text question..."):
                try:
                    payload = {"question": text_question, "k": k_text}
                    response = requests.post(f"{BACKEND_URL}/ask", json=payload)
                    if response.status_code == 200:
                        res = response.json()
                        option = extract_option(res.get("predicted_answer", "No answer returned"))
                        st.success("Answer generated!")
                        st.subheader("Predicted Answer Option:")
                        st.write(option)
                        st.subheader("Confidence:")
                        st.write(f'{res.get("confidence", 0)*100:.1f}%')
                    else:
                        st.error("Error processing text question!")
                except Exception as e:
                    st.error(f"Failed to get answer: {str(e)}")
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()
