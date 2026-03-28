"""
backend/text_correction.py
──────────────────────────
Sửa lỗi nhận dạng giọng nói tiếng Việt cho thuật ngữ vật lý THPT.
Hybrid approach:
  Bước 1 — Regex word-boundary: fix lỗi cố định, KHÔNG đụng từ đúng
  Bước 2 — LLM: chỉ sửa lỗi ngữ nghĩa/ngữ cảnh còn lại
"""

import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_groq: Groq | None = None

def _get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


# ══════════════════════════════════════════════════════════════════
# BƯỚC 1 — REGEX RULES
#
# Dùng re.sub với word-boundary (\b) để KHÔNG đụng vào từ đúng
# bên trong từ khác.
#
# Format: (pattern_regex, replacement)
# Thứ tự: cụm DÀI trước, từ đơn sau — tránh partial match.
#
# CHUẨN ĐƠN VỊ: theo prompts.py TTS tiếng Việt
#   C (culông) → "culông"  KHÔNG phải "Coulomb"
#   N (niutơn) → "niutơn"  KHÔNG phải "Newton"
#   J (jun)    → "jun"
# ══════════════════════════════════════════════════════════════════

_RULES: list[tuple[str, str]] = [
     # ── SỬA KÝ HIỆU TOÁN HỌC (LLM vẫn viết) ──────────────────────────
    (r"(?<![a-zA-Z0-9])=(?![a-zA-Z0-9])",   " bằng "),  # dấu = đứng độc lập
    (r"×",                                   " nhân "),
    (r"\*",                                  " nhân "),
    (r"(?<![a-zA-Z0-9])/(?![a-zA-Z0-9])",    " chia "),  # dấu / đứng độc lập
    (r"\+",                                  " cộng "),
    (r"(?<![a-zA-Z0-9])-(?![a-zA-Z0-9])",    " trừ "),   # dấu - đứng độc lập
    
    # ── SỬA ĐƠN VỊ (ưu tiên trước) ────────────────────────────────────
    (r"m/s²",                   "mét trên giây bình phương"),
    (r"m/s2",                   "mét trên giây bình phương"),
    (r"km/h",                   "kilômét trên giờ"),
    (r"km/h",                   "kilômét trên giờ"),
    (r"\bkg\b",                 "kilôgam"),
    (r"\bN\b",                  "niutơn"),
    (r"\bJ\b",                  "jun"),
    (r"\bW\b",                  "oát"),
    (r"\bHz\b",                 "héc"),
    (r"\bPa\b",                 "pascal"),
    (r"\bC\b",                  "culông"),
    (r"\bV\b",                  "vôn"),
    (r"\bΩ\b",                  "ôm"),
    (r"\bT\b",                  "tesla"),
    (r"\bg\b",                  "g"),  # giữ nguyên, không sửa vì có thể là gia tốc
    
    # ── SỬA LŨY THỪA DẠNG CHỮ (x², x³) ──────────────────────────────
    (r"([a-zA-Z0-9]+)²",        r"\1 bình phương"),
    (r"([a-zA-Z0-9]+)³",        r"\1 lập phương"),
    
    # ── SỬA CĂN BẬC HAI ──────────────────────────────────────────────
    (r"√([a-zA-Z0-9]+)",        r"căn bậc hai của \1"),
    (r"sqrt\(([^)]+)\)",        r"căn bậc hai của \1"),
    
    # ── SỬA KÝ HIỆU HY LẠP ───────────────────────────────────────────
    (r"π",                      "pi"),
    (r"λ",                      "lăm-đa"),
    (r"ω",                      "ô-mê-ga"),
    (r"Δ",                      "biến thiên"),
    (r"(?<![a-zA-Z])μ(?![a-zA-Z])", "muy"),  # μ đứng độc lập

    # ── LŨY THỪA DẠNG ^ ────────────────────────────────────────────
    # Phải xử lý TRƯỚC các rule khác vì chứa số
    (r"10\^-12",            "mười mũ trừ mười hai"),
    (r"10\^-9",             "mười mũ trừ chín"),
    (r"10\^-6",             "mười mũ trừ sáu"),
    (r"10\^-3",             "mười mũ trừ ba"),
    (r"10\^3",              "mười mũ ba"),
    (r"10\^6",              "mười mũ sáu"),
    (r"10\^9",              "mười mũ chín"),
    (r"\bx\^2\b",           "x bình phương"),
    (r"\bx\^3\b",           "x lập phương"),
    (r"\bv\^2\b",           "v bình phương"),
    (r"\br\^2\b",           "r bình phương"),

    # ── ĐƠN VỊ / PHÉP ĐO — cụm dài trước ──────────────────────────
    (r"\btrên dây bình\b",          "trên giây bình phương"),
    (r"\btrên dây vuông\b",         "trên giây bình phương"),
    (r"\bmét trên dây bình\b",      "mét trên giây bình phương"),
    (r"\btrên dây\b",               "trên giây"),     # "m/s" bị đọc nhầm
    (r"\bmét trên dây\b",           "mét trên giây"),
    (r"\bkm trên dây\b",            "km trên giây"),
    (r"\bki lô gam\b",              "kilôgam"),
    (r"\bki lo gam\b",              "kilôgam"),
    (r"\bki lô mét\b",              "kilômét"),
    (r"\bmicro cu lông\b",          "micrôculông"),
    (r"\bmicro cu lom\b",           "micrôculông"),
    (r"\bnano cu lông\b",           "nanoculông"),
    (r"\bnano cu lon\b",            "nanoculông"),
    (r"\bmili am pe\b",             "miliampe"),
    (r"\bkilo ôm\b",                "kilôôm"),
    (r"\bmê ga ôm\b",               "mêgaôm"),

    # ── TÊN NHÀ KHOA HỌC / ĐỊNH LUẬT / ĐƠN VỊ TTS CHUẨN ──────────
    # QUAN TRỌNG: dùng cách đọc TTS theo prompts.py, KHÔNG dùng tên tiếng Anh
    (r"\bcu lôm\b",                 "culông"),   # C → "culông" theo TTS
    (r"\bcu lom\b",                 "culông"),
    (r"\bcu lông\b",                "culông"),   # FIX TEST 5
    (r"\bniu tơn\b",                "niutơn"),   # N → "niutơn" theo TTS
    (r"\bniu ton\b",                "niutơn"),
    (r"\bam pe\b",                  "ampe"),
    (r"\bpát can\b",                "pascal"),
    (r"\bpa can\b",                 "pascal"),
    (r"\bhen ri\b",                 "henry"),
    (r"\bte la\b",                  "tesla"),
    (r"\bfa ra\b",                  "fara"),
    (r"\bđốp lơ\b",                 "Doppler"),
    (r"\bđốp le\b",                 "Doppler"),
    (r"\bai n xtanh\b",             "Einstein"),
    (r"\banh xtanh\b",              "Einstein"),

    # ── SQRT — TIẾNG ANH BỊ GIỮ NGUYÊN ────────────────────────────
    (r"\bsqrt\b",                   "căn bậc hai"),   # FIX TEST 4

    # ── THUẬT NGỮ ĐỘNG HỌC ─────────────────────────────────────────
    (r"\bnem ngang\b",              "ném ngang"),
    (r"\bnem xiên\b",               "ném xiên"),
    (r"\bdộng học\b",               "động học"),
    (r"\bđọng học\b",               "động học"),
    (r"\bdộng lực\b",               "động lực"),
    (r"\bđọng lực\b",               "động lực"),
    (r"\bmặt phẳng nghiên\b",       "mặt phẳng nghiêng"),
    (r"\bmặt phẳng ngan\b",         "mặt phẳng ngang"),  # FIX TEST 2
    (r"\bquán đường\b",             "quãng đường"),
    (r"\bgia tốt\b",                "gia tốc"),
    # QUAN TRỌNG: KHÔNG có rule "vận tố" → "vận tốc"
    # vì "vận tốc" chứa substring "vận tố" → bị replace thành "vận tốcc"  ← BUG GỐC
    (r"\bvận tốt\b",                "vận tốc"),   # chỉ match "vận tốt" chính xác
    (r"\btần sổ\b",                 "tần số"),
    (r"\bbước xóng\b",              "bước sóng"),
    (r"\bsống ngang\b",             "sóng ngang"),
    (r"\bsống dọc\b",               "sóng dọc"),
    (r"\bsống âm\b",                "sóng âm"),
    (r"\bsống điện từ\b",           "sóng điện từ"),
    (r"\bđao động\b",               "dao động"),
    (r"\bcon lắt lò xo\b",          "con lắc lò xo"),
    (r"\bcon lắt\b",                "con lắc"),

    # ── LỰC / MA SÁT ───────────────────────────────────────────────
    (r"\bhệ số ma fast\b",          "hệ số ma sát"),   # FIX TEST 2
    (r"\bhệ số ma fát\b",           "hệ số ma sát"),
    (r"\bma fast\b",                "ma sát"),
    (r"\bma fát\b",                 "ma sát"),
    (r"\bma xát\b",                 "ma sát"),
    (r"\bmassat\b",                 "ma sát"),

    # ── ĐIỆN HỌC ────────────────────────────────────────────────────
    (r"\bhiệu điện thể\b",          "hiệu điện thế"),
    (r"\bđiện thể\b",               "điện thế"),
    (r"\bmạch r l c\b",             "mạch RLC"),
    (r"\bmạch rlc\b",               "mạch RLC"),

    # ── SỐ LIỆU THƯỜNG GẶP ─────────────────────────────────────────
    (r"\bg bằng 10\b",  "g bằng 10 mét trên giây bình phương"),
    (r"\bg bằng 9,8\b", "g bằng 9 phẩy 8 mét trên giây bình phương"),
    (r"\bk bằng 9\b",   "k bằng 9 nhân mười mũ chín"),

    # ── TIẾNG ANH KHÁC BỊ GIỮ NGUYÊN ──────────────────────────────
    (r"\bdelta\b",      "biến thiên"),
    (r"\bomega\b",      "tần số góc"),
    (r"\blambda\b",     "bước sóng"),
    (r"\bepsilon\b",    "suất điện động"),
    (r"\bgamma\b",      "tia gamma"),
    (r"\btheta\b",      "góc theta"),
]

# Compile sẵn để tăng tốc
_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _RULES
]


# ── PROMPT CHO LLM FALLBACK ─────────────────────────────────────
_LLM_PROMPT = """Bạn là trợ lý sửa lỗi nhận dạng giọng nói tiếng Việt cho câu hỏi vật lý THPT.

NHIỆM VỤ: Sửa lỗi phiên âm/nhận dạng sai trong câu — KHÔNG thêm, KHÔNG giải thích, KHÔNG thay đổi nội dung.

QUY TẮC BẮT BUỘC:
- Chỉ sửa từ bị nhận dạng SAI do âm thanh gần giống
- KHÔNG sửa từ đã đúng (ví dụ: "vận tốc" đã đúng thì GIỮ NGUYÊN)
- KHÔNG thêm đơn vị hay thông tin mới
- Trả về ĐÚNG 1 câu đã sửa, không có giải thích

VÍ DỤ ĐÚNG:
  Input:  "Vật có khối lượng 2kg trượt trên mặt phẳng ngan"
  Output: "Vật có khối lượng 2kg trượt trên mặt phẳng ngang"

VÍ DỤ SAI (KHÔNG làm thế này):
  Input:  "tính vận tốc đầu 20m trên giây"
  Output: "tính vận tốcc đầu 20m trên giây"  ← TUYỆT ĐỐI KHÔNG thêm ký tự vào từ đúng"""


# ══════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════

def correct_physics_text(text: str, use_llm: bool = True) -> str:
    """
    Sửa lỗi nhận dạng giọng nói cho văn bản vật lý.

    Args:
        text:    Chuỗi text thô từ Whisper STT
        use_llm: True = dùng LLM fallback sau regex (mặc định)
                 False = chỉ dùng regex (nhanh hơn, offline)

    Returns:
        Chuỗi đã được sửa
    """
    if not text or not text.strip():
        return text

    # ── Bước 1: Regex rules ─────────────────────────────────────
    result = text
    for pattern, replacement in _COMPILED:
        result = pattern.sub(replacement, result)

    # Capitalize chữ đầu câu sau regex
    if result:
        result = result[0].upper() + result[1:]

    # ── Bước 2: LLM fallback ────────────────────────────────────
    if use_llm:
        result = _llm_correct(result)

    return result


def _llm_correct(text: str) -> str:
    """Gọi LLM để sửa lỗi ngữ nghĩa/ngữ cảnh còn lại sau regex."""
    try:
        r = _get_groq().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _LLM_PROMPT},
                {"role": "user",   "content": text},
            ],
            max_tokens=300,
            temperature=0.0,   # deterministic — không sáng tạo
        )
        corrected = r.choices[0].message.content.strip()
        # Sanity check: nếu LLM trả về quá khác (>2x độ dài) thì giữ nguyên
        if len(corrected) > len(text) * 2:
            return text
        return corrected
    except Exception as e:
        print(f"[text_correction] LLM fallback lỗi: {e}")
        return text  # fallback an toàn: giữ nguyên sau regex


def log_correction(original: str, corrected: str) -> None:
    """In log nếu có thay đổi — dùng để debug và bổ sung rules."""
    if original.strip().lower() != corrected.strip().lower():
        print(f"[correction] '{original}' → '{corrected}'")


# ══════════════════════════════════════════════════════════════════
# TEST — python backend/text_correction.py
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        # (input, expected)
        (
            "Ném ngang từ độ cao 80m với vận tốc đầu 20m trên dây",
            "Ném ngang từ độ cao 80m với vận tốc đầu 20m trên giây",
        ),
        (
            "Vật có hệ số ma fast bằng 0,3 trên mặt phẳng ngan",
            "Vật có hệ số ma sát bằng 0,3 trên mặt phẳng ngang",
        ),
        (
            "2 điện tích Q1 Q2 đặt cách nhau 3cm tính lực cu lôm",
            "2 điện tích Q1 Q2 đặt cách nhau 3cm tính lực culông",
        ),
        (
            "sqrt của 82,6 bằng 9,1",
            "Căn bậc hai của 82,6 bằng 9,1",
        ),
        (
            "10^-9 cu lông",
            "Mười mũ trừ chín culông",
        ),
    ]

    # Test chỉ regex (không gọi API)
    print("=" * 60)
    print("TEST TEXT_CORRECTION.PY  (regex only, no LLM)")
    print("=" * 60)
    all_pass = True
    for i, (inp, expected) in enumerate(test_cases, 1):
        result = correct_physics_text(inp, use_llm=False)
        ok = result.strip().lower() == expected.strip().lower()
        status = "✓" if ok else "✗"
        if not ok:
            all_pass = False
        print(f"\n[Test {i}] {status}")
        print(f"  Input   : {inp}")
        print(f"  Output  : {result}")
        print(f"  Expected: {expected}")

    print("\n" + "=" * 60)
    print("KẾT QUẢ:", "TẤT CẢ PASS ✓" if all_pass else "CÓ LỖI ✗")
    print("=" * 60)