import time
import threading
import asyncio
import sys
import io
import os
import re
import numpy as np
import sounddevice as sd
from queue import Queue
from rich.console import Console
import edge_tts
import pygame
from groq import Groq
from dotenv import load_dotenv
from backend.prompts import PHYSBOT_SYSTEM_PROMPT, CORRECTION_ADDON, TTS_RULES, VOICE_INPUT_ADDON
from backend.text_correction import correct_physics_text, log_correction
from backend.rag_pipeline import retrieve_context
load_dotenv()
console = Console()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pygame.mixer.init()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — ghép theo thứ tự: TTS rules → nhân vật → voice → correction → độ dài
# ══════════════════════════════════════════════════════════════════

FULL_SYSTEM_PROMPT = (
    TTS_RULES
    + PHYSBOT_SYSTEM_PROMPT
    + VOICE_INPUT_ADDON
    + CORRECTION_ADDON
    + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUY TẮC ĐỘ DÀI — BẮT BUỘC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Câu lý thuyết: TỐI ĐA 100 từ
- Câu tính toán: TỐI ĐA 150 từ
- KHÔNG viết dài dòng, KHÔNG giải thích lan man
- Trả lời đủ ý, súc tích, vào thẳng vấn đề
"""
)


# ══════════════════════════════════════════════════════════════════
# DETECT BÀI TẬP SỐ — để bật Chain-of-Thought
# ══════════════════════════════════════════════════════════════════

_CALC_KEYWORDS = [
    "tính", "tìm", "bằng bao nhiêu", "bao nhiêu",
    "cho biết", "cho g", "vận tốc", "gia tốc", "quãng đường",
    "thời gian", "lực", "khối lượng", "điện tích", "điện trở",
    "hiệu điện thế", "công suất", "nhiệt lượng", "độ cao",
    "góc", "độ dài", "chu kỳ", "tần số", "bước sóng",
]

def _is_calculation_problem(text: str) -> bool:
    t = text.lower()
    has_digit = bool(re.search(r'\d', t))
    has_keyword = any(kw in t for kw in _CALC_KEYWORDS)
    return has_digit and has_keyword

def _build_user_message(text: str) -> str:
    """Thêm Chain-of-Thought prefix ẩn nếu là bài tập tính toán."""
    if not _is_calculation_problem(text):
        return text

    cot_prefix = (
        "[Hướng dẫn nội bộ — KHÔNG đọc phần này ra loa]: "
        "Đây là bài tập tính toán. "
        "Trước khi trả lời, xác định đúng dạng bài, "
        "dùng đúng công thức SGK, thay số từng bước, kiểm tra đơn vị. "
        "Nếu là ném ngang/xiên: tách 2 phương. "
        "Nếu có ma sát nghiêng: a = g(sinα − μcosα). "
        "Nếu Coulomb: nhân Q1×Q2. "
        "Bắt đầu giải:\n\n"
    )
    return cot_prefix + text


# ══════════════════════════════════════════════════════════════════
# RECORD AUDIO — tự dừng khi im lặng
# ══════════════════════════════════════════════════════════════════

def record_audio(stop_event, data_queue, silence_threshold=0.01, silence_duration=3.0):
    sample_rate = 16000
    chunk_duration = 0.5
    silent_chunks = 0
    started = False

    def callback(indata, frames, time_info, status):
        nonlocal silent_chunks, started

        if status:
            console.print(f"[dim]{status}[/dim]")

        data_queue.put(bytes(indata))

        audio_np = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.abs(audio_np).mean()

        if not started:
            if energy > silence_threshold:
                started = True
                console.print("[dim]Đã phát hiện giọng nói...[/dim]")
            return

        if energy < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks >= silence_duration / chunk_duration:
            console.print(f"[dim]Im lặng {silence_duration}s, dừng ghi...[/dim]")
            stop_event.set()

    # Xóa queue cũ trước khi ghi
    while not data_queue.empty():
        data_queue.get()

    with sd.RawInputStream(
        samplerate=sample_rate,
        dtype="int16",
        channels=1,
        callback=callback,
        blocksize=int(sample_rate * chunk_duration)
    ):
        console.print(f"[dim]Đang nghe... (tự dừng sau {silence_duration}s im lặng)[/dim]")
        while not stop_event.is_set():
            time.sleep(0.1)


# ══════════════════════════════════════════════════════════════════
# TRANSCRIBE — Whisper qua Groq
# ══════════════════════════════════════════════════════════════════

def transcribe(audio_np: np.ndarray) -> str:
    try:
        # Khuếch đại nếu quá nhỏ
        energy = np.abs(audio_np).mean()
        if energy < 0.05:
            gain = min(0.05 / (energy + 1e-9), 10.0)
            audio_np = np.clip(audio_np * gain, -1.0, 1.0)
            console.print(f"[dim]Khuếch đại x{gain:.1f}[/dim]")

        duration = len(audio_np) / 16000
        if duration < 0.8:
            return ""

        import soundfile as sf
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np.astype(np.float32), 16000)
            with open(f.name, "rb") as af:
                result = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=af,
                    language="vi"
                )
        return result.text.strip()
    except Exception as e:
        console.print(f"[red]STT lỗi: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# LLM RESPONSE — có RAG + CoT + retry
# ══════════════════════════════════════════════════════════════════

def get_llm_response(text: str, max_retries: int = 3) -> str:
    console.print(f"[green]RAG query: {text}[/green]")
    # ── RAG: tìm context SGK ─────────────────────────────────────
    context = retrieve_context(text)
    if context:
        console.print(f"[dim]📚 RAG: {len(context)} ký tự context[/dim]")
    else:
        console.print("[dim]📚 RAG: không tìm được context[/dim]")

    # ── Xây dựng user message ─────────────────────────────────────
    # FIX: dùng user_msg (có CoT prefix) thay vì text gốc
    user_msg = _build_user_message(text)

    if context:
        enhanced_msg = (
            f"[TRÍCH ĐOẠN TỪ SGK VẬT LÝ]:\n{context}\n\n"
            f"[CÂU HỎI]:\n{user_msg}"
        )
    else:
        enhanced_msg = user_msg

    # ── Gọi LLM với retry ────────────────────────────────────────
    for attempt in range(max_retries):
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": FULL_SYSTEM_PROMPT},
                    {"role": "user",   "content": enhanced_msg},
                ],
                max_tokens=250,
                temperature=0.7,
                timeout=45.0
            )
            return r.choices[0].message.content

        except Exception as e:
            err = str(e).lower()
            console.print(f"[red]Lần {attempt+1}/{max_retries} thất bại: {e}")

            if "rate_limit" in err:
                wait = 15
            elif "timeout" in err:
                wait = 1
            elif "network" in err or "connection" in err:
                wait = 3
            else:
                wait = 2

            if attempt < max_retries - 1:
                console.print(f"[dim]Chờ {wait}s rồi thử lại...[/dim]")
                time.sleep(wait)

    return "Tui bị lỗi kết nối rồi, bạn thử lại sau nha!"


# ══════════════════════════════════════════════════════════════════
# TTS — Edge-TTS + pygame
# ══════════════════════════════════════════════════════════════════

def analyze_emotion(text: str) -> dict:
    if any(w in text for w in ['!', 'tuyệt', 'hay lắm', 'ngon', 'đỉnh', 'thú vị']):
        return {"rate": "+10%", "volume": "+5%"}
    if any(w in text for w in ['bước một', 'bước hai', 'khoai', 'chi tiết', 'phức tạp']):
        return {"rate": "-10%", "volume": "+0%"}
    return {"rate": "+0%", "volume": "+0%"}


async def speak(text: str):
    emotion = analyze_emotion(text)
    comm = edge_tts.Communicate(
        text,
        voice="vi-VN-HoaiMyNeural",
        rate=emotion["rate"],
        volume=emotion["volume"]
    )
    audio = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]

    pygame.mixer.music.load(io.BytesIO(audio))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    time.sleep(0.5)


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    console.print("[cyan]PhysBot sẵn sàng!")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Nhấn Enter để bắt đầu nói, tự dừng sau 3 giây im lặng.")
    console.print("[cyan]Ctrl+C để thoát.\n")

    try:
        while True:
            console.input("Nhấn Enter để bắt đầu ghi âm...")

            stop_event = threading.Event()
            data_queue = Queue()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue, 0.01, 3.0),
            )
            recording_thread.start()
            recording_thread.join()

            # Gom audio từ queue
            chunks = []
            while not data_queue.empty():
                chunks.append(data_queue.get())
            audio_data = b"".join(chunks)

            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            if audio_np.size == 0:
                console.print("[red]Không nghe thấy gì. Kiểm tra lại mic.")
                continue

            # ── STT ──────────────────────────────────────────────
            with console.status("Đang xử lý...", spinner="dots"):
                t0 = time.time()
                raw_text = transcribe(audio_np)
                t1 = time.time()

            # ── Text correction — CHỈ chạy trên input STT ────────
            # KHÔNG chạy correct_physics_text trên output LLM
            text = correct_physics_text(raw_text)
            log_correction(raw_text, text)

            console.print(f"[yellow]Bạn (raw) : {raw_text}")
            if text != raw_text:
                console.print(f"[yellow]Bạn (fixed): {text}")
            console.print(f"[dim]STT: {t1-t0:.2f}s[/dim]")

            if not text.strip():
                console.print("[red]Không nhận ra giọng nói, thử lại nhé!")
                continue

            # ── LLM ──────────────────────────────────────────────
            with console.status("Đang suy nghĩ...", spinner="dots"):
                t2 = time.time()
                response = get_llm_response(text)
                t3 = time.time()

            # Rút gọn nếu quá dài (bảo vệ TTS)
            if len(response) > 1000:
                response = response[:1000] + "... (tui rút gọn để đọc nhanh hơn nha)"
                console.print("[dim]Đã rút gọn response do quá dài[/dim]")

            console.print(f"[cyan]PhysBot: {response}")
            console.print(f"[dim]LLM: {t3-t2:.2f}s[/dim]")

            # ── TTS ───────────────────────────────────────────────
            t4 = time.time()
            asyncio.run(speak(response))
            t5 = time.time()

            console.print(f"[dim]TTS: {t5-t4:.2f}s | Tổng: {t5-t0:.2f}s[/dim]")
            console.print("")

    except KeyboardInterrupt:
        console.print("\n[red]Thoát...")
