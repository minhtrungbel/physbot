# scripts/ingest.py
"""
Đọc tất cả file PDF trong thư mục temp_pdfs/
Trích xuất text, cắt chunks, lưu vào ChromaDB
"""

import os
import sys
import re
from pathlib import Path

# Thêm đường dẫn gốc để import backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# ========== CẤU HÌNH ==========
PDF_FOLDER = "temp_pdfs"           # Thư mục chứa PDF đã sync từ MEGA
CHROMA_PATH = "chroma_db"          # Thư mục lưu ChromaDB
CHUNK_SIZE = 800                    # Kích thước mỗi chunk (ký tự)

# ========== KHỞI TẠO CHROMA DB ==========
print("🔧 Đang khởi tạo ChromaDB...")

# Tạo client persistent
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Dùng DefaultEmbeddingFunction (chạy ONNX, không cần PyTorch)
from chromadb.utils import embedding_functions
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-MiniLM-L3-v2"
)
# Thêm dòng này để tránh lỗi SSL trên Colab
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Tạo collection (nếu chưa có)
collection = client.get_or_create_collection(
    name="sgk_vatly",
    embedding_function=embedding_fn
)

print(f"✅ ChromaDB sẵn sàng. Collection hiện có: {collection.count()} chunks\n")


# ========== HÀM XỬ LÝ PDF ==========
def extract_text_from_pdf(pdf_path):
    """Đọc text từ file PDF"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"   📄 Số trang: {num_pages}")
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                print(f"   ⚠️ Trang {page_num} không có text (có thể là hình ảnh)")
        
        return text
    except Exception as e:
        print(f"   ❌ Lỗi đọc PDF: {e}")
        return None


def clean_text(text):
    """Làm sạch text: loại bỏ ký tự thừa"""
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    # Loại bỏ các dòng chỉ toàn số trang
    text = re.sub(r'\b\d+\b\s*', '', text)
    return text.strip()


def split_text(text, chunk_size=CHUNK_SIZE):
    """Cắt text thành các chunks theo câu"""
    # Tách thành các câu
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Nếu không cắt được thành câu, cắt theo độ dài
    if not chunks and len(text) > chunk_size:
        for i in range(0, len(text), chunk_size - 100):
            chunks.append(text[i:i + chunk_size])
    
    return chunks


def is_useful_content(text):
    """Kiểm tra xem chunk có phải nội dung học tập không"""
    keywords = [
        'dao động', 'điều hòa', 'biên độ', 'tần số', 'chu kỳ',
        'lực', 'khối lượng', 'gia tốc', 'vận tốc', 'quãng đường',
        'điện tích', 'điện trở', 'công suất', 'năng lượng',
        'phương pháp giải', 'lời giải chi tiết', 'đáp án',
        'công thức', 'định luật', 'bài tập'
    ]
    
    text_lower = text.lower()
    count = sum(1 for kw in keywords if kw in text_lower)
    
    # Chunks có ít nhất 2 từ khóa và dài hơn 50 ký tự
    return count >= 2 and len(text) > 50


# ========== HÀM INGEST CHÍNH ==========
def ingest_all_pdfs():
    """Đọc tất cả PDF trong folder và ingest vào ChromaDB"""
    
    # Kiểm tra thư mục PDF
    pdf_folder = Path(PDF_FOLDER)
    if not pdf_folder.exists():
        print(f"❌ Thư mục {PDF_FOLDER} không tồn tại!")
        print("   Hãy tạo thư mục và đặt file PDF vào đó.")
        return
    
    # Lấy tất cả file PDF
    pdf_files = list(pdf_folder.rglob("*.pdf")) + list(pdf_folder.rglob("*.PDF"))

    pdf_files = list(set(pdf_files))
    
    if not pdf_files:
        print(f"📂 Thư mục {PDF_FOLDER} không có file PDF nào!")
        print("   Hãy upload PDF lên MEGA để tự động sync vào thư mục này.")
        return
    
    print(f"📚 Tìm thấy {len(pdf_files)} file PDF:")
    for f in pdf_files:
        print(f"   - {f.name}")
    
    print("\n" + "="*60)
    print("🚀 BẮT ĐẦU INGEST")
    print("="*60)
    
    total_chunks = 0
    
    for pdf_path in pdf_files:
        print(f"\n📄 Đang xử lý: {pdf_path.name}")
        
        # Đọc text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"   ⚠️ Bỏ qua: không đọc được text")
            continue
        
        # Làm sạch text
        text = clean_text(text)
        
        # Cắt chunks
        chunks = split_text(text)
        print(f"   ✂️ Cắt thành {len(chunks)} chunks")
        
        # Lọc chunks có nội dung hữu ích
        useful_chunks = [c for c in chunks if is_useful_content(c)]
        print(f"   📌 Giữ lại {len(useful_chunks)} chunks có nội dung học tập")
        
        # Thêm vào ChromaDB
        if useful_chunks:
            for i, chunk in enumerate(useful_chunks):
                try:
                    collection.add(
                        documents=[chunk],
                        metadatas=[{
                            "source": pdf_path.name,
                            "chunk_index": i
                        }],
                        ids=[f"{pdf_path.stem}_{i}"]
                    )
                except Exception as e:
                    print(f"   ❌ Lỗi khi thêm chunk {i}: {e}")
            
            total_chunks += len(useful_chunks)
            print(f"   ✅ Đã thêm {len(useful_chunks)} chunks vào ChromaDB")
        else:
            print(f"   ⚠️ Không có chunk hữu ích nào, bỏ qua")
    
    # Kết quả
    print("\n" + "="*60)
    print("✅ INGEST HOÀN TẤT!")
    print(f"   📊 Tổng số file: {len(pdf_files)}")
    print(f"   📦 Tổng số chunks: {total_chunks}")
    print(f"   💾 ChromaDB lưu tại: {os.path.abspath(CHROMA_PATH)}")
    print(f"   📚 Collection hiện có: {collection.count()} chunks")
    print("="*60)


# ========== MAIN ==========
if __name__ == "__main__":
    ingest_all_pdfs()
