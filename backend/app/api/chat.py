from fastapi import APIRouter
from app.models.schemas import ChatRequest
from app.services.embedding_service import embed_query
from app.services.vector_store import search_vectors
from app.services.llm_service import generate_answer

router = APIRouter(prefix="/chat")

@router.post("/")
async def chat(req: ChatRequest):
    query_embedding = embed_query(req.question)
    context = search_vectors(query_embedding)
    answer = generate_answer(req.question, context)
    return {"answer": answer}
