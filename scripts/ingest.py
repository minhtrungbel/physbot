# scripts/ingest.py
import os
import glob
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"  # sửa lại
# ── CẤU HÌNH ─────────────────────────────────────────────────────
PDF_DIR   = "data/raw"          # để PDF SGK vào đây
DB_DIR    = "data/chroma_db"    # ChromaDB sẽ lưu ở đây
CHUNK_SIZE = 500                # ký tự mỗi chunk
OVERLAP    = 50                 # overlap giữa các chunk

# ── MODEL EMBEDDING ───────────────────────────────────────────────
# MiniLM nhẹ, chạy được trên PC không cần GPU
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Thử PyPDF2 trước, nếu không có text thì dùng OCR."""
    # Thử text-based trước
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        if len(text.strip()) > 100:  # có text thật
            return text
    except:
        pass

    # Fallback: OCR từng trang
    print(f"  → Dùng OCR (chậm hơn ~30s/trang)...")
    text = ""
    images = convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH)
    for i, img in enumerate(images):
        page_text = pytesseract.image_to_string(img, lang="vie")
        text += page_text + "\n"
        if (i + 1) % 10 == 0:
            print(f"    OCR xong trang {i+1}/{len(images)}")
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Cắt text thành chunks có overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest():
    # Tạo thư mục nếu chưa có
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Khởi tạo ChromaDB
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Xóa collection cũ nếu có (để chạy lại sạch)
    try:
        client.delete_collection("physbot_sgk")
    except:
        pass
    
    collection = client.create_collection(
        name="physbot_sgk",
        metadata={"description": "SGK Vật lý 10-11-12"}
    )
    
    # Tìm tất cả PDF trong data/raw/
    pdf_files = glob.glob(f"{PDF_DIR}/**/*.pdf", recursive=True) + \
                glob.glob(f"{PDF_DIR}/*.pdf")
    
    if not pdf_files:
        print(f"Không tìm thấy PDF trong {PDF_DIR}/")
        print("Tải SGK từ sach.giaoducvietnam.vn rồi để vào data/raw/")
        return
    
    print(f"Tìm thấy {len(pdf_files)} file PDF")
    
    all_chunks = []
    all_ids = []
    all_metadata = []
    
    for pdf_path in pdf_files:
        print(f"Đang xử lý: {pdf_path}")
        
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"  → Không đọc được text (PDF scan?), bỏ qua")
            continue
            
        chunks = chunk_text(text)
        filename = Path(pdf_path).stem
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # bỏ chunk quá ngắn
                continue
            chunk_id = f"{filename}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({"source": filename, "chunk_index": i})
    
    if not all_chunks:
        print("Không có chunk nào để nhúng!")
        return
    
    print(f"\nĐang nhúng {len(all_chunks)} chunks vào ChromaDB...")
    print("(Lần đầu có thể mất 5-15 phút tùy số lượng PDF)")
    
    # Nhúng theo batch 100 để không out of memory
    BATCH = 100
    for i in range(0, len(all_chunks), BATCH):
        batch_chunks    = all_chunks[i:i+BATCH]
        batch_ids       = all_ids[i:i+BATCH]
        batch_meta      = all_metadata[i:i+BATCH]
        batch_embeddings = model.encode(batch_chunks).tolist()
        
        collection.add(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_meta
        )
        print(f"  Xong {min(i+BATCH, len(all_chunks))}/{len(all_chunks)} chunks")
    
    print(f"\n✓ Xong! Đã nhúng {len(all_chunks)} chunks vào {DB_DIR}/")

if __name__ == "__main__":
    ingest()
