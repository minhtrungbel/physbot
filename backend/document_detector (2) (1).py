"""
backend/document_detector.py
─────────────────────────────
Xác nhận ảnh có chứa tài liệu / sách / văn bản hay không.

Pipeline 2 tầng:
  Tầng 1 — OpenCV (nhanh, offline, ~50ms):
    - Kiểm tra tỉ lệ hình chữ nhật (sách/tờ giấy thường ~A4)
    - Kiểm tra % pixel trắng/sáng (trang giấy sáng hơn nền)
    - Phát hiện đường thẳng ngang (dòng chữ tạo ra lines)
    → Nếu PASS cả 3 tiêu chí: "Có thể là tài liệu" → lên Tầng 2

  Tầng 2 — Groq Vision (chính xác, ~1-2s):
    - Hỏi thẳng: "Có phải sách/tài liệu/văn bản không?"
    - Trả về True/False + mô tả ngắn để TTS

Cách dùng:
    from backend.document_detector import detect_document, DocumentResult
    result = detect_document(img_bytes, groq_client)
    if result.is_document:
        # chụp thật
    else:
        speak(result.guidance)   # hướng dẫn người mù điều chỉnh
"""

import io
import re
import base64
from dataclasses import dataclass

import numpy as np


# ══════════════════════════════════════════════════════════════════
# CONFIG — có thể override qua .env
# ══════════════════════════════════════════════════════════════════

# Tầng 1 — OpenCV
MIN_RECT_AREA_RATIO   = 0.15   # vật thể chiếm ít nhất 15% frame
MIN_WHITE_RATIO       = 0.07   # ít nhất 7% pixel sáng trong vùng vật thể
MAX_ASPECT_DEVIATION  = 2.5    # tỉ lệ w/h tối đa (A4: ~1.41, sách dày ~0.7-2.0)
MIN_HLINE_COUNT       = 3      # tối thiểu 3 đường ngang (dòng chữ)

# Tầng 2 — Vision
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VISION_FALLBACK = "meta-llama/llama-4-maverick-17b-128e-instruct"


# ══════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════

@dataclass
class DocumentResult:
    is_document: bool          # True = xác nhận là tài liệu, chụp được
    confidence: str            # "high" / "medium" / "low"
    guidance: str              # câu TTS hướng dẫn nếu chưa phải tài liệu
    opencv_score: dict         # debug: chi tiết các tiêu chí OpenCV
    vision_response: str       # debug: raw response từ Vision API


# ══════════════════════════════════════════════════════════════════
# TẦNG 1 — OPENCV PREFILTER
# ══════════════════════════════════════════════════════════════════

def _opencv_precheck(img_bytes: bytes) -> tuple[bool, dict]:
    """
    Kiểm tra nhanh bằng OpenCV.
    Trả về (passed, score_dict).

    Tiêu chí:
    1. rect_ok   : contour lớn nhất có diện tích đủ lớn và tỉ lệ hợp lý
    2. white_ok  : vùng bên trong sáng (trang giấy)
    3. hline_ok  : có đường ngang (dòng chữ) → dùng HoughLinesP
    """
    score = {
        "rect_ok":    False,
        "white_ok":   False,
        "hline_ok":   False,
        "area_ratio": 0.0,
        "white_ratio": 0.0,
        "hline_count": 0,
        "aspect":     0.0,
    }

    try:
        import cv2

        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, score

        frame_h, frame_w = img.shape[:2]
        frame_area = frame_h * frame_w

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)

        # ── Tiêu chí 1: Tìm vật thể hình chữ nhật lớn ──────────
        _, thresh = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest   = max(contours, key=cv2.contourArea)
            area      = cv2.contourArea(largest)
            area_ratio = area / frame_area
            score["area_ratio"] = round(area_ratio, 3)

            x, y, w, h = cv2.boundingRect(largest)
            aspect = w / h if h > 0 else 0
            score["aspect"] = round(aspect, 2)

            rect_ok = (
                area_ratio >= MIN_RECT_AREA_RATIO
                and 0.4 <= aspect <= MAX_ASPECT_DEVIATION
            )
            score["rect_ok"] = rect_ok

            # ── Tiêu chí 2: % pixel sáng trong bounding rect ────
            if rect_ok:
                roi        = gray[y:y+h, x:x+w]
                white_pix  = np.sum(roi > 180)
                white_ratio = white_pix / (w * h) if w * h > 0 else 0
                score["white_ratio"] = round(float(white_ratio), 3)
                score["white_ok"]    = white_ratio >= MIN_WHITE_RATIO

        # ── Tiêu chí 3: Đường ngang (dòng chữ) bằng HoughLinesP ─
        # Edge detect trước để HoughLines dễ tìm hơn
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=frame_w * 0.25,   # tối thiểu 25% chiều rộng frame
            maxLineGap=20
        )

        hline_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Đường gần nằm ngang (±20° để nhận giấy nghiêng)
                if angle <= 20 or angle >= 160:
                    hline_count += 1

        score["hline_count"] = hline_count
        score["hline_ok"]    = hline_count >= MIN_HLINE_COUNT

        # Pass nếu ít nhất 2/3 tiêu chí đạt
        passed_count = sum([score["rect_ok"], score["white_ok"], score["hline_ok"]])
        passed = passed_count >= 2
        return passed, score

    except ImportError:
        # OpenCV không có → bỏ qua tầng 1, để Vision quyết định
        return True, score
    except Exception as e:
        return True, score   # lỗi không chặn → cho qua Vision


# ══════════════════════════════════════════════════════════════════
# TẦNG 2 — GROQ VISION
# ══════════════════════════════════════════════════════════════════

_VISION_PROMPT = """Nhìn vào ảnh này và trả lời 2 điều:

1. Có tài liệu/sách/tờ giấy/văn bản nào trong ảnh không?
2. Nếu có: có thể đọc được nội dung không? (đủ sáng, thấy chữ, không bị che quá 55%)

LƯU Ý QUAN TRỌNG:
- Người dùng là người khiếm thị nên tay cầm sách là BÌNH THƯỜNG → vẫn READY: YES nếu thấy rõ nội dung
- Chỉ READY: NO khi: ảnh quá tối, tờ giấy bị che >55%, hoàn toàn không thấy chữ
- Giấy/sách nghiêng một chút vẫn OK

Trả lời ĐÚNG format này (không thêm gì khác):
IS_DOC: YES hoặc NO
READY: YES hoặc NO
REASON: [1 câu ngắn bằng tiếng Việt mô tả vấn đề nếu NO, hoặc "Tài liệu rõ ràng" nếu YES]

Ví dụ trả lời khi OK (kể cả có tay cầm):
IS_DOC: YES
READY: YES
REASON: Tài liệu rõ ràng

Ví dụ trả lời khi chưa OK:
IS_DOC: YES
READY: NO
REASON: Ảnh quá tối, không thấy chữ

Ví dụ khi không có tài liệu:
IS_DOC: NO
READY: NO
REASON: Chỉ thấy bàn và nền, chưa có sách hay tờ giấy nào"""


def _call_vision(groq_client, img_bytes: bytes) -> str:
    """Gọi Groq Vision, thử fallback model nếu lỗi."""
    img_b64 = base64.b64encode(img_bytes).decode()
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            },
            {"type": "text", "text": _VISION_PROMPT}
        ]
    }]

    for model in [VISION_MODEL, VISION_FALLBACK]:
        try:
            resp = groq_client.chat.completions.create(
                model=model,
                max_tokens=120,
                temperature=0,
                messages=messages,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            continue

    return f"ERROR: {last_err}"


def _parse_vision_response(raw: str) -> tuple[bool, bool, str]:
    """
    Parse response Vision API.
    Trả về (is_doc, is_ready, reason).
    """
    is_doc  = bool(re.search(r"IS_DOC:\s*YES", raw, re.IGNORECASE))
    is_ready = bool(re.search(r"READY:\s*YES", raw, re.IGNORECASE))
    reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE)
    reason = reason_match.group(1).strip() if reason_match else "Không xác định được"
    return is_doc, is_ready, reason


# ══════════════════════════════════════════════════════════════════
# TẠO CÂU HƯỚNG DẪN TTS
# ══════════════════════════════════════════════════════════════════

def _build_guidance(is_doc: bool, is_ready: bool, reason: str,
                    opencv_score: dict) -> str:
    """
    Tạo câu TTS hướng dẫn phù hợp cho người mù.
    Ưu tiên thông tin cụ thể từ Vision reason.
    """
    if is_doc and is_ready:
        return "Tài liệu đã sẵn sàng, tui chụp nhé!"

    if not is_doc:
        # Không phải tài liệu — hướng dẫn cụ thể
        if opencv_score.get("white_ratio", 0) < 0.1:
            return (
                "Tui chưa thấy sách hay tờ giấy nào trước camera. "
                "Bạn đặt tài liệu vào chính giữa, cách camera khoảng năm mươi xăng-ti-mét nhé."
            )
        return (
            f"Tui chưa nhận ra tài liệu. {reason}. "
            "Bạn thử đặt sách hoặc tờ giấy vào giữa khung hình nhé."
        )

    # Là tài liệu nhưng chưa ready
    # Dùng reason từ Vision (đã bằng tiếng Việt)
    return f"Gần rồi! {reason}. Bạn điều chỉnh lại một chút nhé."


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def detect_document(img_bytes: bytes, groq_client) -> DocumentResult:
    """
    Xác nhận ảnh có chứa tài liệu/sách/văn bản không.

    Args:
        img_bytes    : JPEG bytes từ camera
        groq_client  : Groq client đã khởi tạo

    Returns:
        DocumentResult với:
          .is_document  → True = chụp được, False = cần điều chỉnh
          .guidance     → câu TTS hướng dẫn người mù
          .confidence   → "high" / "medium" / "low"
    """
    # ── Tầng 1: OpenCV precheck ──────────────────────────────────
    opencv_passed, opencv_score = _opencv_precheck(img_bytes)

    # Nếu OpenCV đã reject rõ ràng (0/3 tiêu chí) → skip Vision, trả về ngay
    passed_count = sum([
        opencv_score.get("rect_ok", False),
        opencv_score.get("white_ok", False),
        opencv_score.get("hline_ok", False),
    ])

    if passed_count == 0:
        guidance = (
            "Tui chưa thấy tài liệu nào trước camera. "
            "Bạn đặt sách hoặc tờ giấy vào giữa, "
            "cách camera khoảng năm mươi xăng-ti-mét nhé."
        )
        return DocumentResult(
            is_document=False,
            confidence="high",
            guidance=guidance,
            opencv_score=opencv_score,
            vision_response="SKIPPED (OpenCV 0/3)",
        )

    # ── Tầng 2: Vision xác nhận ──────────────────────────────────
    vision_raw = _call_vision(groq_client, img_bytes)
    is_doc, is_ready, reason = _parse_vision_response(vision_raw)

    # Tổng hợp confidence
    if opencv_passed and is_doc and is_ready:
        confidence = "high"
    elif is_doc and is_ready:
        confidence = "medium"      # Vision OK nhưng OpenCV chỉ 1/3
    elif is_doc and not is_ready:
        confidence = "medium"
    else:
        confidence = "low"

    is_document = is_doc and is_ready
    guidance    = _build_guidance(is_doc, is_ready, reason, opencv_score)

    return DocumentResult(
        is_document=is_document,
        confidence=confidence,
        guidance=guidance,
        opencv_score=opencv_score,
        vision_response=vision_raw,
    )
