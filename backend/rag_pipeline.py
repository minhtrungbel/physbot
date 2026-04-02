import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "data/chroma_db"
COLLECTION_NAME = "physbot_sgk"

TOP_K = 8
FINAL_TOP_K = 3
MAX_DISTANCE = 35

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def detect_query_type(question: str):
    q = question.lower()

    if any(x in q for x in ["giải", "tính", "tìm", "bao nhiêu"]):
        return "exercise"

    return "theory"


def rerank_chunks(question: str, docs, metas):
    scored = []

    q_words = question.lower().split()

    for doc, meta in zip(docs, metas):
        score = 0
        text = doc.lower()

        for word in q_words:
            if word in text:
                score += 2

        if detect_query_type(question) == "exercise":
            if "lời giải" in text or "bài" in text:
                score += 3

        scored.append((score, doc, meta))

    scored.sort(reverse=True, key=lambda x: x[0])

    top = scored[:FINAL_TOP_K]

    return [x[1] for x in top], [x[2] for x in top]


def merge_neighbor_chunks(collection, meta):
    source = meta.get("source")
    idx = meta.get("chunk_index", 0)

    merged = []

    for offset in [-1, 0, 1]:
        try:
            neighbor_id = f"{source}_{idx + offset}"

            result = collection.get(ids=[neighbor_id])

            if result["documents"]:
                merged.append(result["documents"][0])

        except:
            continue

    return "\n".join(merged)


def retrieve_context(question: str):
    try:
        model = _get_model()
        collection = _get_collection()

        embedding = model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=TOP_K,
            include=["documents", "distances", "metadatas"]
        )

        docs = results["documents"][0]
        dists = results["distances"][0]
        metas = results["metadatas"][0]

        filtered_docs = []
        filtered_metas = []

        for doc, dist, meta in zip(docs, dists, metas):
            if dist <= MAX_DISTANCE:
                filtered_docs.append(doc)
                filtered_metas.append(meta)

        if not filtered_docs:
            return ""

        docs, metas = rerank_chunks(
            question,
            filtered_docs,
            filtered_metas
        )

        expanded = []

        for meta in metas:
            merged = merge_neighbor_chunks(collection, meta)
            if merged:
                expanded.append(merged)

        return "\n\n---\n\n".join(expanded)

    except Exception as e:
        print(f"[RAG ERROR] {e}")
        return ""
def build_rag_prompt(question: str, context: str) -> str:
    if not context:
        return question

    return f"""
Đây là tài liệu tham khảo liên quan từ SGK và sách bài tập Vật lý.

Ưu tiên:
- dùng đúng thông tin trong tài liệu
- tổng hợp các ý liên quan
- tránh lặp lại
- trả lời đúng trọng tâm câu hỏi

Tài liệu:
{context}

Câu hỏi:
{question}
"""
