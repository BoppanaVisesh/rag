from fastapi import APIRouter, UploadFile, File
import os
from app.config import UPLOAD_DIR
from app.services.document_loader import load_document
from app.utils.text_splitter import split_text
from app.services.embedding_service import create_embeddings
from app.services.vector_store import store_embeddings

router = APIRouter(prefix="/upload")

@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    text = load_document(path)
    chunks = split_text(text)
    embeddings = create_embeddings(chunks)
    store_embeddings(chunks, embeddings)
    return {"status": "uploaded"}
