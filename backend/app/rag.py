# backend/app/rag.py
import os
import numpy as np
import faiss
import openai

class JEE_RAG:
    def __init__(self):
        self.qa_data = []  # List of dicts: {"question": str, "answer": str}
        self.embeddings = None
        self.index = None
        self.dimension = 1536  # For OpenAI's text-embedding-ada-002
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def initialize_from_pdf(self, qa_data):
        """Initialize vector database with QA pairs."""
        self.qa_data = qa_data
        texts = [item["question"] for item in qa_data]
        embeddings = [self.get_openai_embedding(text) for text in texts]    
        self.embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)
    
    def get_openai_embedding(self, text: str):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        return np.array(embedding, dtype="float32")
    
    def get_answer(self, query: str, k: int):
        if self.index is None or not self.qa_data:
            return {"error": "No data available. Please load the PDFs first."}
        # Generate query embedding
        query_embedding = self.get_openai_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        supporting = []
        for i, idx in enumerate(indices[0]):
            q_item = self.qa_data[idx]
            supporting.append({
                "text": q_item["question"],
                "answer": q_item["answer"],
                "distance": float(distances[0][i])
            })
        # Build a prompt that instructs the model to perform calculations and output only the answer option.
        prompt = "Based on the following reference questions and their correct answers:\n"
        for item in supporting:
            prompt += f"Question: {item['text']}\nAnswer: {item['answer']}\n\n"
        prompt += (
            "Now, answer the following question by performing all necessary calculations and provide "
            "only the correct answer option number (1, 2, 3, or 4) as your final output:\n"
            f"Question: {query}\n"
            "Answer option:"
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant who performs detailed calculations and returns only the final answer option number (1, 2, 3, or 4) without any extra text."},
            {"role": "user", "content": prompt}
        ]
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=10,
            temperature=0.0,
        )
        answer_text = completion.choices[0].message["content"].strip()
        # Extract the final option number (expecting one of: 1, 2, 3, 4)
        import re
        match = re.search(r'\b[1-4]\b', answer_text)
        final_answer = match.group(0) if match else answer_text
        confidence = 0.95  # Example confidence value; adjust as needed.
        return {
            "predicted_answer": final_answer,
            "confidence": confidence,
            "supporting_questions": supporting
        }

