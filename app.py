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
from backend.prompts import PHYSBOT_SYSTEM_PROMPT, CORRECTION_ADDON,TTS_RULES,VOICE_INPUT_ADDON
from backend.text_correction import correct_physics_text, log_correction

load_dotenv()
console = Console()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pygame.mixer.init()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

FULL_SYSTEM_PROMPT = TTS_RULES + PHYSBOT_SYSTEM_PROMPT + VOICE_INPUT_ADDON + CORRECTION_ADDON

# Thêm quy tắc trả lời ngắn gọn
FULL_SYSTEM_PROMPT += """
⚠️ QUY TẮC ĐỘ DÀI TUYỆT ĐỐI ⚠️
- Câu lý thuyết: TỐI ĐA 100 từ (khoảng 5-7 dòng)
- Câu tính toán: TỐI ĐA 150 từ (khoảng 8-10 dòng)
- KHÔNG viết dài dòng, KHÔNG giải thích lan man
- Trả lời đủ ý, súc tích, vào thẳng vấn đề
"""
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
    if not _is_calculation_problem(text):
        return text

    cot_prefix = (
        "[Hướng dẫn nội bộ — KHÔNG đọc phần này ra loa]: "
        "Đây là bài tập tính toán. "
        "Trước khi trả lời, tui phải xác định đúng dạng bài, "
        "dùng đúng công thức SGK, thay số từng bước, kiểm tra đơn vị. "
        "Nếu là ném ngang/xiên: tách 2 phương. "
        "Nếu có ma sát nghiêng: a = g(sinα − μcosα). "
        "Nếu Coulomb: nhân Q1×Q2. "
        "Bắt đầu giải:\n\n"
    )
    return cot_prefix + text


def record_audio(stop_event, data_queue, silence_threshold=0.01, silence_duration=3.0):
    """
    Ghi âm, tự động dừng khi im lặng quá silence_duration giây
    """
    sample_rate = 16000
    chunk_duration = 0.5  # 500ms mỗi chunk để tính năng lượng
    silent_chunks = 0
    started = False  # đã bắt đầu có âm thanh chưa
    
    def callback(indata, frames, time, status):
        nonlocal silent_chunks, started
        
        if status:
            console.print(f"[dim]{status}[/dim]")
        
        # Lưu audio vào queue
        data_queue.put(bytes(indata))
        
        # Tính năng lượng
        audio_np = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.abs(audio_np).mean()
        
        # Nếu chưa bắt đầu nói, chờ có âm thanh mới tính
        if not started:
            if energy > silence_threshold:
                started = True
                console.print("[dim]Đã phát hiện giọng nói...[/dim]")
            return
        
        # Đã bắt đầu nói, kiểm tra im lặng
        if energy < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
        
        # Nếu im lặng đủ lâu, dừng ghi
        if silent_chunks >= silence_duration / chunk_duration:
            console.print(f"[dim]Im lặng {silence_duration}s, dừng ghi...[/dim]")
            stop_event.set()
    
    # Reset queue
    while not data_queue.empty():
        data_queue.get()
    
    with sd.RawInputStream(
        samplerate=sample_rate,
        dtype="int16",
        channels=1,
        callback=callback,
        blocksize=int(sample_rate * chunk_duration)
    ):
        console.print(f"[dim]Đang nghe... (sẽ tự dừng sau {silence_duration}s im lặng)[/dim]")
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    try:
        energy = np.abs(audio_np).mean()
        if energy < 0.05:
            gain = 0.05 / (energy + 1e-9)
            gain = min(gain, 10.0)
            audio_np = audio_np * gain
            audio_np = np.clip(audio_np, -1.0, 1.0)
            console.print(f"[dim]Khuech dai x{gain:.1f}[/dim]")

        duration = len(audio_np) / 16000
        if duration < 0.8:
            return ""

        import soundfile as sf, tempfile
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
        console.print(f"[red]STT loi: {e}")
        return ""


def get_llm_response(text: str, max_retries=3) -> str:
    user_msg = _build_user_message(text)
    
    for attempt in range(max_retries):
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": FULL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=250,
                temperature=0.7,
                timeout=15.0
            )
            return r.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e).lower()
            console.print(f"[red]Lần {attempt+1}/{max_retries} thất bại: {e}")
            
            if "rate_limit" in error_msg:
                wait_time = 15
                console.print(f"[dim]Rate limit, chờ {wait_time}s...[/dim]")
            elif "timeout" in error_msg:
                wait_time = 1
                console.print("[dim]Timeout, thử lại ngay...[/dim]")
            elif "network" in error_msg or "connection" in error_msg:
                wait_time = 3
                console.print(f"[dim]Lỗi mạng, chờ {wait_time}s...[/dim]")
            else:
                wait_time = 2
            
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                return "Tui bị lỗi kết nối rồi, bạn thử lại sau nha!"
    
    return "Tui bị lỗi kết nối rồi, bạn thử lại sau nha!"
def analyze_emotion(text: str) -> dict:
    if any(w in text for w in ['!', 'tuyệt', 'hay lắm','ngon', 'đỉnh', 'thú vị']):
        return {"rate": "+10%", "volume": "+5%"}
    if any(w in text for w in ['bước một', 'bước hai','khoai', 'chi tiết', 'phức tạp']):
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


if __name__ == "__main__":
    console.print("[cyan]PhysBot san sang!")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Nhan Enter de bat dau noi, noi xong se tu dung sau 3 giay im lang.")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    try:
        while True:
            console.input("Nhan Enter de bat dau ghi am...")
            
            stop_event = threading.Event()
            data_queue = Queue()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue, 0.01, 3.0),
            )
            recording_thread.start()
            
            # Chờ cho đến khi stop_event được set (tự động sau 3s im lặng)
            recording_thread.join()
            
            # Lấy dữ liệu audio từ queue
            chunks = []
            while not data_queue.empty():
                chunks.append(data_queue.get())
            audio_data = b"".join(chunks)
            
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            
            if audio_np.size > 0:
                with console.status("Dang xu ly...", spinner="dots"):
                    t0 = time.time()
                    raw_text = transcribe(audio_np)
                    t1 = time.time()
                
                # Sửa lỗi chính tả từ giọng nói
                text = correct_physics_text(raw_text)
                log_correction(raw_text, text)
                
                console.print(f"[yellow]Ban (raw) : {raw_text}")
                if text != raw_text:
                    console.print(f"[yellow]Ban (fixed): {text}")
                console.print(f"[dim]STT: {t1-t0:.2f}s[/dim]")
                
                if not text.strip():
                    console.print("[red]Khong nhan ra giong noi, thu lai nhe!")
                    continue
                
                with console.status("Dang suy nghi...", spinner="dots"):
                    t2 = time.time()
                    response = get_llm_response(text)
                    t3 = time.time()
                response = correct_physics_text(response)

                # Cắt response nếu quá dài để TTS nhanh hơn
                if len(response) > 1000:
                    response = response[:1000] + "\n\n(Tui rút gọn để đọc nhanh hơn nha)"
                    console.print("[dim]Đã rút gọn response do quá dài[/dim]")
                
                console.print(f"[cyan]PhysBot: {response}")
                console.print(f"[dim]LLM: {t3-t2:.2f}s[/dim]")
                
                t4 = time.time()
                asyncio.run(speak(response))
                t5 = time.time()
                
                console.print(f"[dim]TTS: {t5-t4:.2f}s | Tong: {t5-t0:.2f}s[/dim]")
                console.print("")
                
            else:
                console.print("[red]Khong nghe thay gi. Kiem tra lai mic.")
                
    except KeyboardInterrupt:
        console.print("\n[red]Thoat...")