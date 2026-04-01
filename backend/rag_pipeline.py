# backend/rag_pipeline.py
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "data/chroma_db"
TOP_K = 4
MAX_DISTANCE = 500.0   # nới threshold

# Load 1 lần khi import — không load lại mỗi câu hỏi
_model: SentenceTransformer | None = None
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection("physbot_sgk")
    return _collection


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _model


def retrieve_context(question: str) -> str:
    """
    Tìm các đoạn SGK liên quan đến câu hỏi.
    Trả về string context để nhét vào prompt.
    """
    try:
        model = _get_model()
        collection = _get_collection()

        # Nhúng câu hỏi
        query_embedding = model.encode(question).tolist()

        # Tìm kiếm
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "distances", "metadatas"]
        )

        docs = results["documents"][0]
        dists = results["distances"][0]
        metas = results["metadatas"][0]

        print(f"[RAG] Query: {question}")
        print(f"[RAG] Found {len(docs)} chunks")
        print(f"[RAG] Distances: {dists}")

        if not docs:
            return ""

        context_parts = []

        for doc, dist, meta in zip(docs, dists, metas):
            if dist <= MAX_DISTANCE:
                source = meta.get("source", "SGK")
                context_parts.append(
                    f"[{source}] (dist={dist:.3f})\n{doc.strip()}"
                )

        if not context_parts:
            print("[RAG] Không có chunk nào qua threshold")
            return ""

        print(f"[RAG] Dùng {len(context_parts)} chunks")
        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        print(f"[rag_pipeline] Lỗi: {e}")
        return ""


def build_rag_prompt(question: str, context: str) -> str:
    """
    Tạo prompt có nhúng context SGK.
    Dùng thay cho question thuần khi gọi LLM.
    """
    if not context:
        return question

    return f"""Dưới đây là tài liệu tham khảo từ sách giáo khoa Vật lý:

{context}

---

Dựa vào tài liệu trên (nếu liên quan), hãy trả lời câu hỏi sau:
{question}"""   
