import os
import glob
import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from pdf2image import convert_from_path
from dotenv import load_dotenv
import pytesseract
import pdfplumber

load_dotenv()

PDF_DIRS      = ["data/raw", "data/exercises"]
PROCESSED_DIR = "data/processed"
DB_DIR        = "data/chroma_db"

COLLECTION_NAME = "physbot_sgk"
POPPLER_PATH    = r"C:\poppler-25.12.0\Library\bin"
TESSERACT_PATH  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def is_vietnamese_text(text: str) -> bool:
    vietnamese_chars = set(
        "àáâãèéêìíòóôõùúýăđơư"
        "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
        "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ"
        "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ"
    )
    count = sum(1 for c in text if c in vietnamese_chars)
    return count / max(len(text), 1) >= 0.01


def semantic_chunk_text(text: str):
    parts = re.split(r"(Bài\s+\d+|Câu\s+\d+)", text)
    chunks = []
    current = ""

    for part in parts:
        if re.match(r"(Bài\s+\d+|Câu\s+\d+)", part):
            if current.strip():
                chunks.append(current.strip())
            current = part
        else:
            current += "\n" + part

    if current.strip():
        chunks.append(current.strip())

    return chunks


def ocr_with_tesseract(image) -> str:
    return pytesseract.image_to_string(image, lang="vie")


def extract_text_from_pdf(pdf_path: str) -> str:
    filename   = Path(pdf_path).stem
    cache_path = Path(PROCESSED_DIR) / f"{filename}.txt"

    if cache_path.exists():
        print(f"  [cache] Dùng cache: {cache_path}", flush=True)
        return cache_path.read_text(encoding="utf-8")

    text = ""

    # Bước 1: Thử PyPDF2
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if len(text.strip()) > 100 and is_vietnamese_text(text):
            print("  [PyPDF2] OK, có dấu tiếng Việt.", flush=True)
            cache_path.write_text(text, encoding="utf-8")
            return text
        else:
            print("  [PyPDF2] Thiếu dấu → thử pdfplumber.", flush=True)
            text = ""
    except Exception as e:
        print(f"  [PyPDF2] Lỗi: {e}", flush=True)

    # Bước 2: Thử pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    text += page_text + "\n"

        if len(text.strip()) > 100 and is_vietnamese_text(text):
            print("  [pdfplumber] OK, có ký hiệu toán học.", flush=True)
            cache_path.write_text(text, encoding="utf-8")
            return text
        else:
            print("  [pdfplumber] Vẫn thiếu → chuyển sang OCR.", flush=True)
            text = ""
    except Exception as e:
        print(f"  [pdfplumber] Lỗi: {e}", flush=True)

    # Bước 3: OCR Tesseract
    print("  OCR bằng Tesseract (tiếng Việt)...", flush=True)

    checkpoint_dir = Path(PROCESSED_DIR) / f"{filename}_pages"
    checkpoint_dir.mkdir(exist_ok=True)

    print("  Đang convert PDF sang ảnh (có thể mất vài phút)...", flush=True)
    images = convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH)

    total = len(images)
    print(f"  Convert xong! Tổng số trang: {total}", flush=True)

    for i, img in enumerate(images):
        page_cache = checkpoint_dir / f"page_{i:04d}.txt"

        if page_cache.exists():
            print(f"  Trang {i+1}/{total} [cache]", flush=True)
            continue

        print(f"  OCR trang {i+1}/{total}...", flush=True)
        page_text = ocr_with_tesseract(img)
        page_cache.write_text(page_text, encoding="utf-8")
        print(f"  Trang {i+1}/{total} xong", flush=True)

    all_pages = sorted(checkpoint_dir.glob("page_*.txt"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in all_pages)
    cache_path.write_text(text, encoding="utf-8")

    return text


def ingest():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' đã tồn tại, tiếp tục thêm vào.")
    except Exception:
        collection = client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Tạo mới collection '{COLLECTION_NAME}' với cosine distance.")

    # ── Gom PDF từ tất cả folders, bỏ trùng theo tên file ──
    seen_stems = set()
    pdf_files  = []

    for pdf_dir in PDF_DIRS:
        if not os.path.exists(pdf_dir):
            print(f"Folder không tồn tại, bỏ qua: {pdf_dir}")
            continue
        for path in glob.glob(f"{pdf_dir}/**/*.pdf", recursive=True):
            stem = Path(path).stem
            if stem not in seen_stems:
                seen_stems.add(stem)
                pdf_files.append(path)
            else:
                print(f"  [skip trùng] {Path(path).name}")

    print(f"\nTìm thấy {len(pdf_files)} PDF\n")

    all_chunks = []
    all_ids    = []
    all_meta   = []

    for pdf_path in pdf_files:
        print(f"Đang xử lý: {pdf_path}", flush=True)

        text     = extract_text_from_pdf(pdf_path)
        chunks   = semantic_chunk_text(text)
        filename = Path(pdf_path).stem
        source_folder = "exercises" if "exercises" in pdf_path else "raw"

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue

            all_chunks.append(chunk)
            all_ids.append(f"{filename}_{i}")
            all_meta.append({
                "source":        filename,
                "chunk_index":   i,
                "source_folder": source_folder,
            })

    if not all_chunks:
        print("Không có chunk nào để ingest!")
        return

    print(f"\nĐang tạo embeddings cho {len(all_chunks)} chunks...", flush=True)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_meta
    )

    total = collection.count()
    print(f"\nIngest hoàn tất!")
    print(f"   Chunks vừa thêm : {len(all_chunks)}")
    print(f"   Collection tổng : {total} chunks")


if __name__ == "__main__":
    ingest()
