from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import database
import file_loader
import embedding
import generation_model
import asyncio

app = FastAPI()

# Initialize components
db_connection = database.DatabaseConnection()
embedding_handler = embedding.TextChunkerEmbedder(
    "sentence-transformers/all-MiniLM-L6-v2"
)
rag_system = generation_model.RAGSystem(db_connection, embedding_handler)


# Define input models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/upload/")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Uploads multiple documents, processes embeddings, and stores them."""
    try:
        mapped_documents = {}  # Dictionary to store {filename: text}

        for file in files:
            content = await file.read()
            text = content.decode("utf-8")
            mapped_documents[file.filename] = text

        # Process embeddings for all documents at once
        doc_embeddings = embedding_handler.map_document_embeddings(mapped_documents)

        # Store embeddings in the database (should handle duplicates)
        db_connection.insert_document_embeddings(doc_embeddings)

        return {"message": f"Uploaded {len(files)} document(s) successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/")
async def query_rag(request: QueryRequest):
    """Handles retrieval-augmented generation requests."""
    try:
        answer = rag_system.generate_answer(request.question, request.top_k)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
