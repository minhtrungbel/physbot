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

load_dotenv()

PDF_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DB_DIR = "data/chroma_db"

COLLECTION_NAME = "physbot_sgk"
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
    text = pytesseract.image_to_string(image, lang="vie")
    return text


def extract_text_from_pdf(pdf_path: str):
    filename = Path(pdf_path).stem
    cache_path = Path(PROCESSED_DIR) / f"{filename}.txt"

    if cache_path.exists():
        print(f"  [cache] Dung cache: {cache_path}", flush=True)
        return cache_path.read_text(encoding="utf-8")

    text = ""

    # Buoc 1: Thu extract text bang PyPDF2
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if len(text.strip()) > 100 and is_vietnamese_text(text):
            print("  [PyPDF2] Extract thanh cong, co dau tieng Viet.", flush=True)
            cache_path.write_text(text, encoding="utf-8")
            return text
        else:
            print("  [PyPDF2] Khong co dau hoac qua ngan -> chuyen sang OCR.", flush=True)
            text = ""

    except Exception as e:
        print(f"  [PyPDF2] Loi: {e} -> chuyen sang OCR.", flush=True)

    # Buoc 2: OCR bang Tesseract
    print("  OCR bang Tesseract (tieng Viet)...", flush=True)

    checkpoint_dir = Path(PROCESSED_DIR) / f"{filename}_pages"
    checkpoint_dir.mkdir(exist_ok=True)

    print("  Dang convert PDF sang anh (co the mat vai phut)...", flush=True)
    images = convert_from_path(
        pdf_path,
        dpi=200,
        poppler_path=POPPLER_PATH
    )

    total = len(images)
    print(f"  Convert xong! Tong so trang: {total}", flush=True)

    for i, img in enumerate(images):
        page_cache = checkpoint_dir / f"page_{i:04d}.txt"

        if page_cache.exists():
            print(f"  Trang {i+1}/{total} [cache]", flush=True)
            continue

        print(f"  OCR trang {i+1}/{total}...", flush=True)
        page_text = ocr_with_tesseract(img)
        page_cache.write_text(page_text, encoding="utf-8")
        print(f"  Trang {i+1}/{total} xong", flush=True)

    # Ghep tat ca trang lai
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
    except:
        collection = client.create_collection(COLLECTION_NAME)

    pdf_files = glob.glob(f"{PDF_DIR}/**/*.pdf", recursive=True)

    all_chunks = []
    all_ids = []
    all_meta = []

    for pdf_path in pdf_files:
        print(f"Dang xu ly: {pdf_path}", flush=True)

        text = extract_text_from_pdf(pdf_path)
        chunks = semantic_chunk_text(text)
        filename = Path(pdf_path).stem

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue

            all_chunks.append(chunk)
            all_ids.append(f"{filename}_{i}")
            all_meta.append({
                "source": filename,
                "chunk_index": i
            })

    if not all_chunks:
        print("Khong co chunk nao de ingest!")
        return

    print(f"\nDang tao embeddings cho {len(all_chunks)} chunks...", flush=True)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_meta
    )

    print("Ingest hoan tat!")


if __name__ == "__main__":
    ingest()
