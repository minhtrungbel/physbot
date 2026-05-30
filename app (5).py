"""
app.py  —  PhysBot Pi Client v3.2.0
──────────────────────────────────
THAY ĐỔI v3.2.0 (so với v3.1.0):
  [FIX] plify_mp3() — dùng ffmpeg thay pydub (ổn định hơn trên Pi)
        → ffmpeg sẵn có trên Raspberry Pi OS, không cần pydub/libav
        → fallback giữ nguyên file gốc nếu ffmpeg cũng lỗi
  [FIX] AUDIODEV = "hw:0,0" (Google Voice HAT là card 0)
  [FIX] TTS bổ sung các điểm còn thiếu:
        + capture_image_bytes() trả None → TTS "Camera không phản hồi"
        + img_bytes < 5000 → TTS "Ảnh bị hỏng, thử lại"
        + Attempt đầu (trước doc check) → TTS "Đã thấy sách, đang kiểm tra..."
        + get_response() khi img_bytes None → TTS trước khi return
  [KEEP] Toàn bộ logic v3.1.0 giữ nguyên
"""

# ══════════════════════════════════════════════════════════════════
# PHẢI SET TRƯỚC KHI IMPORT SOUNDDEVICE
# ══════════════════════════════════════════════════════════════════
import os
os.environ["PA_ALSA_PLUGHW"]    = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["SDL_AUDIODRIVER"]   = "alsa"
os.environ["AUDIODEV"]          = "hw:0,0"   # Google Voice HAT = card 0

import time
import threading
import asyncio
import sys
import re
import tempfile
import json
import uuid
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty

warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider'")

import numpy as np
import sounddevice as sd
import httpx
from rich.console import Console
from dotenv import load_dotenv
from groq import Groq

from backend.text_correction import correct_physics_text, log_correction
from backend.document_detector import detect_document
from backend.drive_uploader import upload_image_background, GDRIVE_ENABLED

load_dotenv()
console = Console()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

API_BASE    = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "45"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ══════════════════════════════════════════════════════════════════
# AUDIO CONFIG
# ══════════════════════════════════════════════════════════════════

_dev_raw     = os.getenv("INPUT_DEVICE", "0")
INPUT_DEVICE = int(_dev_raw) if _dev_raw.lstrip("-").isdigit() else _dev_raw

HW_SR       = int(os.getenv("HW_SAMPLERATE", "48000"))
TARGET_SR   = int(os.getenv("TARGET_SR",      "16000"))
HW_CHANNELS = int(os.getenv("HW_CHANNELS",   "2"))

UNIFIED_CHUNK_MS     = 80
UNIFIED_BLOCK_FRAMES = int(HW_SR * UNIFIED_CHUNK_MS / 1000)

STREAM_WARMUP_SEC = float(os.getenv("STREAM_WARMUP_SEC", "3.0"))


def _auto_detect_device() -> int | str:
    try:
        devices = sd.query_devices()
        if not devices:
            console.print("[yellow]⚠ PortAudio không thấy device nào[/yellow]")
            return INPUT_DEVICE

        console.print("[dim]─── Danh sách audio devices ───[/dim]")
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                console.print(f"[dim]  [{i}] {d['name']} "
                               f"(in={d['max_input_channels']} "
                               f"sr={int(d['default_samplerate'])})[/dim]")

        keywords = ["googlevoice", "voicehat", "google voice", "sndrpigoogle", "spdif"]
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                if any(kw in d["name"].lower() for kw in keywords):
                    console.print(f"[green]✓ Tìm thấy Google Voice HAT: [{i}] {d['name']}[/green]")
                    return i

        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                console.print(f"[yellow]⚠ Dùng device [{i}] {d['name']}[/yellow]")
                return i

    except Exception as e:
        console.print(f"[yellow]⚠ Auto-detect lỗi: {e}[/yellow]")

    return INPUT_DEVICE


def _resample(audio_hw: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly
    if audio_hw.dtype == np.int32:
        audio_f = audio_hw.astype(np.float32) / 2147483648.0
    else:
        audio_f = audio_hw.astype(np.float32)
    if audio_f.ndim == 2:
        mono = audio_f.mean(axis=1)
    else:
        mono = audio_f.copy()
    g    = math.gcd(TARGET_SR, HW_SR)
    up   = TARGET_SR // g
    down = HW_SR     // g
    return resample_poly(mono, up, down).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# WAKE WORD CONFIG
# ══════════════════════════════════════════════════════════════════

WAKE_MODEL        = os.getenv("WAKE_MODEL", "hey_jarvis")
WAKE_THRESHOLD    = float(os.getenv("WAKE_THRESHOLD", "0.95"))
ENERGY_MIN        = float(os.getenv("WAKE_ENERGY_MIN", "0.02"))
WAKE_COOLDOWN_SEC = 3.0
POST_TTS_MUTE_SEC = float(os.getenv("POST_TTS_MUTE_SEC", "3.0"))
OWW_CHUNK_SAMPLES = int(TARGET_SR * UNIFIED_CHUNK_MS / 1000)


# ══════════════════════════════════════════════════════════════════
# STATE MACHINE
# ══════════════════════════════════════════════════════════════════

STATE_IDLE   = "IDLE"
STATE_ACTIVE = "ACTIVE"

_bot_state       = STATE_IDLE
_state_lock      = threading.Lock()
_last_active_ts  = 0.0
IDLE_TIMEOUT_SEC = float(os.getenv("IDLE_TIMEOUT_SEC", "30"))
_activated_event = threading.Event()


# ══════════════════════════════════════════════════════════════════
# UNIFIED STREAM
# ══════════════════════════════════════════════════════════════════

_record_queue: Queue       = Queue()
_wake_last_trigger: float  = 0.0
_tts_playing: bool         = False
_last_deactivate_ts: float = 0.0
WAKE_POST_DEACTIVATE_COOLDOWN = float(os.getenv("WAKE_POST_DEACTIVATE_COOLDOWN", "5.0"))

_stream_start_ts: float  = 0.0
_stream_warmed_up: bool  = False

_oww_builtin_model = None
_oww_custom_sess   = None
_oww_custom_iname  = None
_oww_audio_feats   = None
_use_custom_model  = False

_custom_audio_buf: np.ndarray = np.zeros(0, dtype=np.int16)
_unified_stream = None


def _unified_audio_callback(indata: np.ndarray, frames: int, time_info, status):
    global _wake_last_trigger, _custom_audio_buf, _stream_warmed_up

    if _tts_playing:
        return

    if not _stream_warmed_up:
        if time.time() - _stream_start_ts < STREAM_WARMUP_SEC:
            return
        _custom_audio_buf = np.zeros(0, dtype=np.int16)
        _stream_warmed_up = True
        console.print(f"[dim]✓ Warmup xong — bắt đầu lắng nghe wake word[/dim]")

    audio_16k = _resample(indata)

    if _bot_state == STATE_IDLE:
        score = 0.0

        if _use_custom_model and _oww_custom_sess is not None:
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            _custom_audio_buf = np.concatenate([_custom_audio_buf, audio_int16])
            while len(_custom_audio_buf) >= OWW_CHUNK_SAMPLES:
                chunk = _custom_audio_buf[:OWW_CHUNK_SAMPLES]
                _custom_audio_buf = _custom_audio_buf[OWW_CHUNK_SAMPLES:]
                try:
                    _oww_audio_feats(chunk)
                    emb = _oww_audio_feats.get_features(n_feature_frames=1)
                    if emb is not None and len(emb) > 0:
                        raw = _oww_custom_sess.run(
                            None,
                            {_oww_custom_iname: emb[0].reshape(1, -1).astype(np.float32)}
                        )[0][0][0]
                        score = max(score, float(raw))
                except Exception:
                    pass

        elif _oww_builtin_model is not None:
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            try:
                _oww_builtin_model.predict(audio_int16)
            except Exception:
                return
            for _, buf in _oww_builtin_model.prediction_buffer.items():
                if buf:
                    score = max(score, float(buf[-1]))

        if score >= WAKE_THRESHOLD:
            energy = np.abs(audio_16k).mean()
            if energy < 0.01:
                return
            now = time.time()
            if (now - _wake_last_trigger > WAKE_COOLDOWN_SEC
                    and now - _last_deactivate_ts > WAKE_POST_DEACTIVATE_COOLDOWN):
                _wake_last_trigger = now
                console.print(f"[bold green]✓ Wake word! score={score:.3f}[/bold green]")
                threading.Thread(target=_activate_bot, daemon=True).start()

    else:
        _record_queue.put(audio_16k)


def _start_unified_stream():
    global _stream_start_ts, _stream_warmed_up

    device   = _auto_detect_device()
    errors   = []
    candidates = [device]
    for fb in [0, 1, 2]:
        if fb not in candidates:
            candidates.append(fb)

    for dev in candidates:
        try:
            stream = sd.InputStream(
                samplerate=HW_SR,
                channels=HW_CHANNELS,
                dtype="int32",
                blocksize=UNIFIED_BLOCK_FRAMES,
                callback=_unified_audio_callback,
                device=dev,
            )
            stream.start()
            _stream_start_ts  = time.time()
            _stream_warmed_up = False
            console.print(
                f"[cyan]✓ Unified stream: device={dev} "
                f"{HW_SR}Hz/{HW_CHANNELS}ch → {TARGET_SR}Hz mono[/cyan]"
            )
            console.print(f"[dim]  Warmup {STREAM_WARMUP_SEC:.0f}s...[/dim]")
            return stream
        except Exception as e:
            errors.append(f"device={dev}: {e}")
            console.print(f"[yellow]⚠ Stream device={dev} lỗi: {e}[/yellow]")

    console.print("[red]✗ Không mở được audio stream![/red]")
    for err in errors:
        console.print(f"[red]  {err}[/red]")
    return None


# ══════════════════════════════════════════════════════════════════
# SESSION & LOGGING
# ══════════════════════════════════════════════════════════════════

SESSION_ID         = os.getenv("SESSION_ID") or str(uuid.uuid4())[:8]
SESSION_START_TIME = time.time()
console.print(f"[dim]Session ID: {SESSION_ID}[/dim]")

_IMPLICIT_LOG_PATH = Path("logs/implicit_feedback.jsonl")

HF_TOKEN        = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")


def _log_implicit(event: str, **kwargs):
    record = {
        "ts":         datetime.now(timezone.utc).isoformat(),
        "session_id": SESSION_ID,
        "event":      event,
        **kwargs,
    }
    try:
        _IMPLICIT_LOG_PATH.parent.mkdir(exist_ok=True)
        with open(_IMPLICIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        console.print(f"[dim red]Implicit log lỗi: {e}[/dim red]")


def _log_to_hf():
    if not HF_TOKEN or not HF_DATASET_REPO:
        console.print("[dim]HF logging chưa cấu hình (thiếu HF_TOKEN / HF_DATASET_REPO)[/dim]")
        return

    if not _IMPLICIT_LOG_PATH.exists() or _IMPLICIT_LOG_PATH.stat().st_size == 0:
        console.print("[dim]Không có log để upload[/dim]")
        return

    try:
        import base64

        ts_str  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        hf_path = f"logs/{SESSION_ID}_{ts_str}.jsonl"

        with open(_IMPLICIT_LOG_PATH, "rb") as f:
            file_bytes = f.read()

        encoded = base64.b64encode(file_bytes).decode()

        api_url = f"https://huggingface.co/api/datasets/{HF_DATASET_REPO}/commit/main"
        payload = {
            "commit_message": f"log {SESSION_ID} {ts_str}",
            "summary":        f"log {SESSION_ID} {ts_str}",
            "operations": [
                {
                    "operation": "addOrUpdate",
                    "path":      hf_path,
                    "encoding":  "base64",
                    "content":   encoded,
                }
            ],
        }

        with httpx.Client(timeout=30) as client:
            r = client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type":  "application/json",
                },
                json=payload,
            )

        if r.status_code in (200, 201):
            console.print(f"[green]✓ Log uploaded: {HF_DATASET_REPO}/{hf_path}[/green]")
        else:
            console.print(f"[yellow]⚠ HF upload lỗi {r.status_code}: {r.text[:120]}[/yellow]")

    except Exception as e:
        console.print(f"[yellow]⚠ HF upload exception: {e}[/yellow]")


# ══════════════════════════════════════════════════════════════════
# STATE HELPERS
# ══════════════════════════════════════════════════════════════════

_last_response: str = ""
_last_numbers:  str = ""
_last_topic:    str = ""

_REPEAT_TRIGGERS = [
    "nói lại", "đọc lại", "nhắc lại", "lặp lại",
    "không nghe", "nghe không rõ", "cho nghe lại",
    "lại đi", "lại nha", "lại một lần", "nói lại cho",
    "đọc lại cho", "không nghe rõ", "nghe không thấy",
]
_NOT_UNDERSTAND_TRIGGERS = [
    "không hiểu", "chưa hiểu",
    "giải thích lại", "nói dễ hơn", "đơn giản hơn",
    "khó hiểu", "không rõ", "chưa rõ",
]
_KEEP_DATA_TRIGGERS = [
    "giữ nguyên số liệu", "giữ nguyên dữ liệu", "cùng số liệu",
    "số liệu cũ", "số liệu trên", "bài trên", "đề trên",
    "vẫn vậy", "như cũ", "cũng vậy", "vẫn số đó",
    "giữ nguyên", "tương tự vậy", "tương tự đó",
    "cùng bài", "cùng đề", "đề cũ", "bài cũ",
]

def _is_repeat_request(text: str) -> bool:
    return any(t in text.lower() for t in _REPEAT_TRIGGERS)

def _is_not_understand(text: str) -> bool:
    return any(t in text.lower() for t in _NOT_UNDERSTAND_TRIGGERS)

def _is_keep_data_request(text: str) -> bool:
    return any(t in text.lower() for t in _KEEP_DATA_TRIGGERS)

def _extract_topic(text: str) -> str:
    words = re.findall(r'\b\w{3,}\b', text.lower())
    return " ".join(words[:3]) if words else ""

_NUMBER_PATTERN = re.compile(
    r'\d+(?:[,\.]\d+)?'
    r'(?:\s*(?:kilôgam|kilômét|mét|xentimét|milimét|giây|niutơn|jun|oát|'
    r'vôn|ampe|héc|culông|tesla|fara|henry|pascal|ôm|'
    r'kg|km|cm|mm|m|s|N|J|W|V|A|Hz|C|T|F|H|Pa)[\w\s]*)?',
    re.IGNORECASE
)

def _extract_numbers(text: str) -> str:
    matches = _NUMBER_PATTERN.findall(text)
    cleaned = [m.strip() for m in matches if m.strip() and re.search(r'\d', m)]
    return ", ".join(cleaned)


# ══════════════════════════════════════════════════════════════════
# BEEP
# ══════════════════════════════════════════════════════════════════

_pygame_mixer_ready = False

def _init_pygame_mixer():
    global _pygame_mixer_ready
    if _pygame_mixer_ready:
        return True
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        _pygame_mixer_ready = True
        return True
    except Exception as e:
        console.print(f"[dim red]pygame mixer init lỗi: {e}[/dim red]")
        return False


def _beep(freq: float = 880.0, duration: float = 0.25, volume: float = 0.4):
    try:
        import pygame
        if not _init_pygame_mixer():
            return
        sr    = 44100
        n     = int(sr * duration)
        t     = np.linspace(0, duration, n, endpoint=False)
        wave  = np.sin(2 * math.pi * freq * t)
        fade  = int(sr * 0.02)
        ramp  = np.linspace(0, 1, fade)
        wave[:fade]  *= ramp
        wave[-fade:] *= ramp[::-1]
        pcm    = (wave * volume * 32767).astype(np.int16)
        stereo = np.column_stack([pcm, pcm])
        sound  = pygame.sndarray.make_sound(stereo)
        sound.play()
        pygame.time.wait(int(duration * 1000) + 50)
    except ImportError:
        pass
    except Exception as e:
        console.print(f"[dim red]Beep lỗi: {e}[/dim red]")


# ══════════════════════════════════════════════════════════════════
# DETECT MODE
# ══════════════════════════════════════════════════════════════════

_OCR_TRIGGERS = [
    "đọc văn bản", "đọc sách", "đọc chữ",
    "đọc cho tao", "đọc trang này", "chức năng đọc",
    "đọc hộ", "đọc giúp",
    "đọc nội dung", "đọc tờ", "đọc tờ giấy",
    "đọc tài liệu", "đọc đoạn", "đọc thử",
    "chế độ đọc", "scan chữ", "nhận dạng chữ",
]
_SOLVE_IMAGE_TRIGGERS = [
    # nhóm từ khóa gốc
    "giải bài", "bài có hình", "hình vẽ",
    "xem hình", "chụp bài", "giải hình",
    "bài tập này", "đề này",
    "chế độ giải", "giải đề", "giải hình ảnh",
    "bài tập bằng hình", "chụp đề", "giải từ ảnh",
    "nhìn bài", "xem bài", "đọc bài",
    "chụp hình bài", "giải bài tập này",
    # mở rộng — Whisper hay ra các biến thể này
    "giải bằng ảnh", "giải theo ảnh", "giải theo hình",
    "chụp ảnh bài", "chụp ảnh đề", "chụp hình đề",
    "xem ảnh", "nhìn ảnh", "xem đề",
    "bài tập ảnh", "đề ảnh", "ảnh bài",
    "giải từ hình", "dùng hình", "dùng ảnh",
    "bài này", "đề này", "bài đây", "đề đây",
    "chế độ ảnh", "chế độ hình", "chụp và giải",
    "giải câu này", "câu này", "giải giùm",
    "giải giúp", "giải hộ", "xem câu",
    "chụp lên", "chụp đi", "chụp thử",
    # từ khóa đơn — chỉ match khi ngắn hoặc standalone
    "ảnh", "hình",
]

# Từ khóa đơn chỉ match khi câu rất ngắn (≤ 3 từ)
_SOLVE_IMAGE_SHORT_TRIGGERS = {"ảnh", "hình", "chụp"}

def detect_mode(text: str) -> str:
    t = text.lower().strip()
    words = t.split()

    # OCR check trước
    if any(kw in t for kw in _OCR_TRIGGERS):
        return "OCR"

    # SOLVE_IMAGE — long triggers (2+ từ)
    long_triggers = [kw for kw in _SOLVE_IMAGE_TRIGGERS if len(kw.split()) >= 2]
    if any(kw in t for kw in long_triggers):
        return "SOLVE_IMAGE"

    # SOLVE_IMAGE — short triggers chỉ match khi câu ≤ 4 từ
    if len(words) <= 4:
        if any(kw in t for kw in _SOLVE_IMAGE_SHORT_TRIGGERS):
            return "SOLVE_IMAGE"

    return "NORMAL"


# ══════════════════════════════════════════════════════════════════
# CAMERA CONFIG
# ══════════════════════════════════════════════════════════════════

DIST_MIN_CM   = float(os.getenv("DIST_MIN_CM",   "40"))
DIST_MAX_CM   = float(os.getenv("DIST_MAX_CM",   "70"))
DIST_IDEAL_CM = (DIST_MIN_CM + DIST_MAX_CM) / 2   # 55cm
BLUR_MIN_SCORE = float(os.getenv("BLUR_MIN_SCORE", "100"))

OFFSET_THRESHOLD = float(os.getenv("OFFSET_THRESHOLD", "0.15"))


# ══════════════════════════════════════════════════════════════════
# HC-SR04
# ══════════════════════════════════════════════════════════════════

def get_distance_cm() -> float:
    try:
        import RPi.GPIO as GPIO
        TRIG = int(os.getenv("HCSR04_TRIG", "23"))
        ECHO = int(os.getenv("HCSR04_ECHO", "24"))
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        GPIO.output(TRIG, False)
        time.sleep(0.05)
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        timeout = time.time() + 1.0
        start = time.time()
        while GPIO.input(ECHO) == 0:
            start = time.time()
            if time.time() > timeout:
                GPIO.cleanup()
                return -1.0
        end = time.time()
        timeout = time.time() + 1.0
        while GPIO.input(ECHO) == 1:
            end = time.time()
            if time.time() > timeout:
                GPIO.cleanup()
                return -1.0
        GPIO.cleanup()
        return round((end - start) * 17150, 1)
    except ImportError:
        return DIST_IDEAL_CM
    except Exception:
        return -1.0


# ══════════════════════════════════════════════════════════════════
# OPENCV — DETECT LỆCH TRÁI/PHẢI
# ══════════════════════════════════════════════════════════════════

def _detect_book_offset(img_bytes: bytes) -> float:
    try:
        import cv2

        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0

        frame_w  = img.shape[1]
        frame_cx = frame_w / 2.0

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        largest    = max(contours, key=cv2.contourArea)
        area       = cv2.contourArea(largest)
        frame_area = img.shape[0] * img.shape[1]
        if area < frame_area * 0.10:
            return 0.0

        x, y, w, h = cv2.boundingRect(largest)
        obj_cx = x + w / 2.0
        offset = (obj_cx - frame_cx) / frame_w
        console.print(f"[dim]Book offset: {offset:+.3f} "
                       f"(obj_cx={obj_cx:.0f}, frame_cx={frame_cx:.0f})[/dim]")
        return offset

    except ImportError:
        console.print("[yellow]⚠ OpenCV không có — bỏ qua detect lệch[/yellow]")
        return 0.0
    except Exception as e:
        console.print(f"[dim red]detect_book_offset lỗi: {e}[/dim red]")
        return 0.0


# ══════════════════════════════════════════════════════════════════
# CAPTURE IMAGE
# ══════════════════════════════════════════════════════════════════

def _find_camera():
    import cv2
    import numpy as _np

    env_index  = int(os.getenv("CAMERA_INDEX", "0"))
    candidates = [env_index] + [i for i in range(5) if i != env_index]

    if sys.platform == "win32":
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW]
    else:
        backends = [cv2.CAP_ANY]

    for idx in candidates:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            brightness = 0.0
            for i in range(30):
                ret, frame = cap.read()
                if ret and frame is not None:
                    brightness = float(_np.array(frame).mean())
                    if brightness > 5:
                        break
                time.sleep(0.05)
            backend_name = {cv2.CAP_ANY: "ANY", cv2.CAP_DSHOW: "DSHOW"}.get(backend, str(backend))
            if brightness > 5:
                console.print(f"[dim]Webcam OK — index={idx}, backend={backend_name}[/dim]")
                return idx, cap
            else:
                console.print(f"[dim]Bỏ qua index={idx} [{backend_name}] — frame đen[/dim]")
                cap.release()

    return None


def capture_image_bytes() -> bytes | None:
    """
    Chụp frame tốt nhất từ webcam (chọn trong 3 frame).
    Trả về JPEG bytes RAW — chưa enhance.
    Việc enhance được làm riêng trong enhance_for_api() tuỳ mode.
    """
    try:
        import cv2
        result = _find_camera()
        if result is None:
            console.print("[red]Không tìm thấy camera nào (index 0–4)![/red]")
            return None
        cam_index, cap = result

        # Tăng resolution nếu camera hỗ trợ
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        # Flush + chờ camera ổn định exposure
        for _ in range(10):
            cap.read()
        time.sleep(0.8)

        # Chụp 3 frame, chọn frame sắc nét nhất
        best_frame = None
        best_score = -1.0
        for _ in range(3):
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
            time.sleep(0.1)

        cap.release()

        if best_frame is None:
            console.print(f"[red]cap.read() thất bại tại index={cam_index}[/red]")
            return None

        console.print(f"[dim]Best frame sharpness: {best_score:.1f}[/dim]")
        _, buf = cv2.imencode(".jpg", best_frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
        return buf.tobytes()

    except ImportError:
        console.print("[yellow]OpenCV chưa cài. Chạy: pip install opencv-python[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Camera lỗi: {e}[/red]")
        return None


def _crop_document_region(img: np.ndarray) -> np.ndarray:
    """
    Crop vùng tài liệu (tờ giấy sáng) ra khỏi frame.
    Loại bỏ tay cầm, nền tối — giúp Vision model tập trung vào nội dung.
    Fallback: trả nguyên ảnh nếu không tìm thấy vùng hợp lệ.
    """
    import cv2
    h, w = img.shape[:2]
    farea = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < farea * 0.08:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / ch if ch > 0 else 0
        if 0.3 <= aspect <= 3.0:
            valid.append((area, c))

    if not valid:
        return img

    largest = max(valid, key=lambda x: x[0])[1]
    x, y, cw, ch = cv2.boundingRect(largest)
    pad_x = int(w * 0.02)
    pad_y = int(h * 0.02)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)

    cropped = img[y1:y2, x1:x2]
    ratio   = (cw * ch) / farea
    console.print(f"[dim]doc_crop: ({x1},{y1})-({x2},{y2}) ratio={ratio:.2f}[/dim]")
    return cropped


def enhance_for_api(img_bytes: bytes, mode: str = "SOLVE_IMAGE") -> bytes:
    """
    Tối ưu hoá ảnh trước khi gửi lên API tuỳ theo mode:

    OCR         → crop tài liệu + đen trắng adaptive threshold
                  → chữ đen trên nền trắng sạch, máy đọc tốt nhất
    SOLVE_IMAGE → crop tài liệu + color enhanced (CLAHE + unsharp)
                  → LLM nhìn rõ màu sắc, hình vẽ, sơ đồ
    """
    try:
        import cv2

        nparr = np.frombuffer(img_bytes, np.uint8)
        orig  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if orig is None:
            return img_bytes

        # Bước 1: Crop vùng tài liệu (loại tay cầm, nền)
        frame = _crop_document_region(orig)
        ch, cw = frame.shape[:2]

        # Bước 2: Upscale 2× Lanczos
        frame = cv2.resize(frame, (cw * 2, ch * 2),
                           interpolation=cv2.INTER_LANCZOS4)

        # Bước 3: CLAHE — cân bằng sáng tối không đều
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        if mode == "OCR":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Bilateral — khử noise, giữ cạnh chữ
            gray = cv2.bilateralFilter(gray, d=7,
                                       sigmaColor=25, sigmaSpace=25)

            # Unsharp mask
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
            gray = cv2.addWeighted(gray, 2.0, blur, -1.0, 0)

            # Adaptive threshold — blockSize=51 tốt cho ảnh 1280px+
            fh2, fw2 = gray.shape[:2]
            block = max(51, int(fw2 * 0.04) | 1)   # ~4% width, luôn lẻ
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=block,
                C=15,
            )

            # Morphological opening — khử đốm noise nhỏ
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)

            out = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            console.print(
                f"[dim]enhance_for_api: OCR → B&W threshold "
                f"(block={block}, size={fw2}x{fh2})[/dim]"
            )

        else:
            # SOLVE_IMAGE: giữ màu để LLM hiểu hình vẽ / sơ đồ
            frame = cv2.bilateralFilter(frame, d=5,
                                        sigmaColor=25, sigmaSpace=25)
            blur  = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5)
            frame = cv2.addWeighted(frame, 2.2, blur, -1.2, 0)
            frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=12)
            out   = frame
            fh2, fw2 = out.shape[:2]
            console.print(
                f"[dim]enhance_for_api: SOLVE_IMAGE → color "
                f"(size={fw2}x{fh2})[/dim]"
            )

        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 98])
        enhanced = buf.tobytes()
        console.print(
            f"[dim]enhance_for_api: {len(img_bytes)//1024}KB"
            f" → {len(enhanced)//1024}KB[/dim]"
        )
        return enhanced

    except Exception as e:
        console.print(f"[yellow]⚠ enhance_for_api lỗi: {e} — dùng ảnh gốc[/yellow]")
        return img_bytes


def _check_image_quality(img_bytes: bytes) -> bool:
    """Kiểm tra độ sắc nét bằng Laplacian variance."""
    try:
        import cv2
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        console.print(f"[dim]Blur score: {score:.1f} (min={BLUR_MIN_SCORE})[/dim]")
        return score >= BLUR_MIN_SCORE
    except Exception:
        return True


# ══════════════════════════════════════════════════════════════════
# TTS HELPER
# ══════════════════════════════════════════════════════════════════

def _speak_sync(text: str):
    """
    Gọi speak() từ thread thường (blocking — chờ TTS xong mới tiếp tục).
    Dùng trong guide_and_capture() để hướng dẫn tuần tự.
    """
    t = threading.Thread(
        target=lambda: asyncio.run(speak(text)),
        daemon=True,
    )
    t.start()
    t.join()


# ══════════════════════════════════════════════════════════════════
# GUIDE AND CAPTURE — HC-SR04 + OPENCV + FULL TTS
# ══════════════════════════════════════════════════════════════════

def guide_and_capture() -> bytes | None:
    """
    Hướng dẫn người dùng đặt sách đúng vị trí rồi chụp ảnh.

    TTS ở TỪNG bước — người mù biết chính xác chuyện gì đang xảy ra:
      Bước 1 : HC-SR04 đo khoảng cách
               → "Cách X cm, lùi/tiến thêm Y cm"  (nếu sai)
               → im lặng, chụp ngay               (nếu đúng)
      Bước 2 : Chụp ảnh thử
               → "Camera không phản hồi, thử lại"  (nếu None)
               → "Ảnh bị hỏng, thử lại"            (nếu < 5 KB)
      Bước 3 : detect_document() — xác nhận có tài liệu không
               → doc_result.guidance               (nếu chưa phải tài liệu)
               → "Đã thấy tài liệu, đang kiểm tra vị trí..." (nếu OK, lần đầu)
      Bước 4 : OpenCV detect lệch trái/phải
               → "Dịch sách sang phải/trái một chút"
      Bước 5 : Kiểm tra blur
               → "Ảnh hơi mờ, giữ yên tay"
      Bước 6 : Ảnh đạt
               → "Ảnh ổn rồi, đang xử lý nhé!"
    """
    MAX_ATTEMPTS    = 8
    DOC_CHECK_START = 2   # attempt 1 chỉ hướng dẫn khoảng cách
    _doc_confirmed  = False  # tránh TTS "Đã thấy tài liệu" lặp lại

    for attempt in range(MAX_ATTEMPTS):
        console.print(f"[dim]guide_and_capture attempt {attempt+1}/{MAX_ATTEMPTS}[/dim]")

        # ── BƯỚC 1: Đo khoảng cách ──────────────────────────────
        dist = get_distance_cm()
        console.print(f"[dim]Khoảng cách: {dist} cm[/dim]")

        if dist < 0:
            console.print("[yellow]⚠ HC-SR04 lỗi, bỏ qua check khoảng cách[/yellow]")
            _speak_sync("Cảm biến khoảng cách bị lỗi, tui chụp luôn nhé.")
        elif dist > DIST_MAX_CM:
            diff = round(dist - DIST_IDEAL_CM)
            _speak_sync(
                f"Bạn đang cách cam {int(dist)} xăng-ti-mét. "
                f"Vui lòng đưa sách lại gần thêm khoảng {diff} xăng-ti-mét nữa nhé."
            )
            time.sleep(2.0)
            continue
        elif dist < DIST_MIN_CM:
            diff = round(DIST_IDEAL_CM - dist)
            _speak_sync(
                f"Bạn đang cách cam {int(dist)} xăng-ti-mét, hơi gần quá. "
                f"Vui lòng lùi sách ra xa thêm khoảng {diff} xăng-ti-mét nhé."
            )
            time.sleep(2.0)
            continue
        else:
            console.print(f"[green]✓ Khoảng cách OK: {dist}cm[/green]")

        # ── BƯỚC 2: Chụp ảnh thử ─────────────────────────────────
        img_bytes = capture_image_bytes()

        if img_bytes is None:
            # [FIX v3.2.0] TTS thay vì im lặng
            _speak_sync("Camera không phản hồi, đợi tui thử lại một chút nhé.")
            time.sleep(1.5)
            continue

        if len(img_bytes) < 5000:
            # [FIX v3.2.0] TTS thay vì im lặng
            _speak_sync("Ảnh bị hỏng, thử lại nhé.")
            time.sleep(1.0)
            continue

        # ── BƯỚC 3: Xác nhận tài liệu/văn bản ───────────────────
        if attempt >= DOC_CHECK_START - 1:
            console.print("[dim]Đang xác nhận tài liệu...[/dim]")
            doc_result = detect_document(img_bytes, groq_client)

            console.print(
                f"[dim]DocDetect: is_doc={doc_result.is_document} "
                f"conf={doc_result.confidence} "
                f"opencv={doc_result.opencv_score}[/dim]"
            )
            console.print(f"[dim]Vision: {doc_result.vision_response[:80]}[/dim]")

            if not doc_result.is_document:
                _speak_sync(doc_result.guidance)
                time.sleep(2.5)
                continue
            else:
                # [FIX v3.2.0] Thông báo lần đầu xác nhận thấy tài liệu
                if not _doc_confirmed:
                    _speak_sync("Đã thấy tài liệu, đang kiểm tra vị trí.")
                    _doc_confirmed = True
                console.print("[green]✓ Xác nhận: đây là tài liệu![/green]")
        else:
            # Attempt đầu: bỏ qua doc check, chỉ hướng dẫn khoảng cách
            console.print("[dim]Attempt đầu: bỏ qua doc check[/dim]")

        # ── BƯỚC 4: Detect lệch trái/phải ────────────────────────
        offset = _detect_book_offset(img_bytes)

        # offset < 0  → tâm vật lệch sang trái frame → người dùng dịch sách sang PHẢI
        # offset > 0  → tâm vật lệch sang phải frame → người dùng dịch sách sang TRÁI
        if offset < -OFFSET_THRESHOLD:
            shift_pct = int(abs(offset) * 100)
            _speak_sync(
                f"Tài liệu đang lệch sang trái khoảng {shift_pct} phần trăm. "
                "Bạn dịch sách sang phải một chút nhé."
            )
            time.sleep(2.0)
            continue
        elif offset > OFFSET_THRESHOLD:
            shift_pct = int(abs(offset) * 100)
            _speak_sync(
                f"Tài liệu đang lệch sang phải khoảng {shift_pct} phần trăm. "
                "Bạn dịch sách sang trái một chút nhé."
            )
            time.sleep(2.0)
            continue

        # ── BƯỚC 5: Kiểm tra độ sắc nét ──────────────────────────
        if not _check_image_quality(img_bytes):
            _speak_sync("Ảnh hơi mờ, bạn giữ yên tay và thử lại nhé.")
            time.sleep(1.5)
            continue

        # ── BƯỚC 6: Ảnh đạt ──────────────────────────────────────
        console.print("[green]✓ Ảnh đạt chất lượng! Đang xử lý...[/green]")
        _speak_sync("Ảnh ổn rồi, đang xử lý nhé!")

        if GDRIVE_ENABLED:
            upload_image_background(
                img_bytes,
                prefix="physbot_capture",
                session_id=SESSION_ID,
            )
            console.print("[dim]Drive upload: đang chạy background[/dim]")

        return img_bytes

    # Hết attempts
    _speak_sync("Thử nhiều lần rồi, tui gửi ảnh hiện tại lên nhé.")
    last_img = capture_image_bytes()
    if last_img and GDRIVE_ENABLED:
        upload_image_background(last_img, prefix="physbot_fallback", session_id=SESSION_ID)
    return last_img


# ══════════════════════════════════════════════════════════════════
# GỌI API SERVER
# ══════════════════════════════════════════════════════════════════

def call_api_ask(text: str) -> str:
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            r = client.post(
                f"{API_BASE}/ask",
                json={"question": text, "session_id": SESSION_ID},
            )
            r.raise_for_status()
            answer = r.json().get("answer", "")
            if "bó tay" in answer.lower() or "ngoài phạm vi" in answer.lower():
                _log_implicit("out_of_scope", question=text[:80])
            return answer
    except httpx.ConnectError:
        _log_implicit("api_error", error="ConnectError", endpoint="/ask")
        return "Tui không kết nối được server, bạn kiểm tra wifi nha!"
    except httpx.TimeoutException:
        _log_implicit("api_error", error="Timeout", endpoint="/ask")
        return "Server trả lời quá lâu, bạn thử lại nha!"
    except Exception as e:
        _log_implicit("api_error", error=str(e)[:80], endpoint="/ask")
        console.print(f"[red]API /ask lỗi: {e}[/red]")
        return "Có lỗi xảy ra, bạn thử lại sau nha!"


def call_api_ocr(img_bytes: bytes, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                r = client.post(
                    f"{API_BASE}/ocr",
                    files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
                )
                r.raise_for_status()
                return r.json().get("answer", "")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503 and attempt < retries:
                console.print(f"[yellow]Server 503, retry {attempt+1}/{retries}...[/yellow]")
                time.sleep(3)
                continue
            console.print(f"[red]API /ocr HTTP lỗi {e.response.status_code}[/red]")
            return "Server bận, bạn thử lại sau nhé!"
        except Exception as e:
            console.print(f"[red]API /ocr lỗi: {e}[/red]")
            return "Có lỗi xảy ra, bạn thử lại sau nha!"
    return "Server không phản hồi, thử lại sau nhé!"


def call_api_solve_image(img_bytes: bytes, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                r = client.post(
                    f"{API_BASE}/solve_image",
                    files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
                    params={"session_id": SESSION_ID},
                )
                r.raise_for_status()
                return r.json().get("answer", "")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503 and attempt < retries:
                console.print(f"[yellow]Server 503, retry {attempt+1}/{retries}...[/yellow]")
                time.sleep(3)
                continue
            console.print(f"[red]API /solve_image HTTP lỗi {e.response.status_code}[/red]")
            return "Server bận, bạn thử lại sau nhé!"
        except Exception as e:
            console.print(f"[red]API /solve_image lỗi: {e}[/red]")
            return "Có lỗi xảy ra, bạn thử lại sau nha!"
    return "Server không phản hồi, thử lại sau nhé!"


def get_response(text: str) -> str:
    mode = detect_mode(text)
    console.print(f"[dim]Mode: {mode}[/dim]")

    if mode == "NORMAL":
        return call_api_ask(text)

    if mode == "OCR":
        _speak_sync(
            "Tui chuyển sang chế độ đọc văn bản. "
            "Bạn cầm sách hoặc tờ giấy hướng về phía camera, "
            "cách khoảng năm mươi xăng-ti-mét. "
            "Tui sẽ hướng dẫn từng bước nhé!"
        )
    elif mode == "SOLVE_IMAGE":
        _speak_sync(
            "Tui chuyển sang chế độ giải bài từ hình ảnh. "
            "Bạn đặt đề bài hoặc tờ giấy có bài tập hướng về camera, "
            "cách khoảng năm mươi xăng-ti-mét. "
            "Tui sẽ hướng dẫn từng bước nhé!"
        )

    img_bytes = guide_and_capture()

    if img_bytes is None:
        _speak_sync("Tui không chụp được ảnh, bạn thử lại nhé.")
        return "Tui không chụp được ảnh, bạn thử lại nhé."

    # Enhance ảnh tuỳ mode trước khi gửi API
    img_bytes = enhance_for_api(img_bytes, mode=mode)

    if mode == "OCR":
        return call_api_ocr(img_bytes)
    return call_api_solve_image(img_bytes)


# ══════════════════════════════════════════════════════════════════
# TRANSCRIBE
# ══════════════════════════════════════════════════════════════════

def transcribe(audio_np: np.ndarray, language: str = "vi") -> str:
    try:
        import soundfile as sf
        energy = np.abs(audio_np).mean()
        if energy < 0.005:
            return ""
        if energy < 0.05:
            gain     = min(0.05 / (energy + 1e-9), 10.0)
            audio_np = np.clip(audio_np * gain, -1.0, 1.0)
            console.print(f"[dim]Khuếch đại x{gain:.1f}[/dim]")
        if len(audio_np) / TARGET_SR < 0.5:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            sf.write(tmp_path, audio_np.astype(np.float32), TARGET_SR)
            with open(tmp_path, "rb") as af:
                result = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3", file=af, language=language,
                )
        os.unlink(tmp_path)
        return result.text.strip()
    except Exception as e:
        console.print(f"[red]STT lỗi: {e}[/red]")
        return ""


# ══════════════════════════════════════════════════════════════════
# WAKE WORD MODEL LOADING
# ══════════════════════════════════════════════════════════════════

def _load_wake_model():
    global _oww_builtin_model, _oww_custom_sess, _oww_custom_iname
    global _oww_audio_feats, _use_custom_model

    if WAKE_MODEL.endswith(".onnx"):
        model_path = Path(WAKE_MODEL)
        if not model_path.exists():
            console.print(f"[red]Không tìm thấy: {WAKE_MODEL}[/red]")
            return False
        try:
            import onnxruntime as ort
            from openwakeword.utils import AudioFeatures

            sess              = ort.InferenceSession(str(model_path))
            _oww_custom_sess  = sess
            _oww_custom_iname = sess.get_inputs()[0].name
            _oww_audio_feats  = AudioFeatures()
            _use_custom_model = True

            console.print(f"[green]✓ Custom model: {model_path.name}[/green]")
            return True

        except ImportError as e:
            console.print(f"[red]Thiếu thư viện: {e}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Lỗi load custom model: {e}[/red]")
            return False

    try:
        from openwakeword.model import Model
        import openwakeword

        oww_dir    = Path(openwakeword.__file__).parent
        models_dir = oww_dir / "resources" / "models"
        candidates = list(models_dir.glob(f"{WAKE_MODEL}*.onnx"))

        if not candidates:
            available = [f.stem for f in models_dir.glob("*.onnx")
                         if not any(x in f.stem for x in
                                    ["embedding", "melspectrogram", "silero"])]
            console.print(f"[red]Không tìm thấy model '{WAKE_MODEL}'[/red]")
            console.print(f"[yellow]Có sẵn: {available}[/yellow]")
            return False

        model_path = str(sorted(candidates)[-1])
        oww = Model(wakeword_model_paths=[model_path])
        _oww_builtin_model = oww
        _use_custom_model  = False

        console.print(f"[green]✓ Built-in model: {Path(model_path).name}[/green]")
        return True

    except ImportError:
        console.print("[red]openwakeword chưa cài.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Lỗi load built-in model: {e}[/red]")
        return False


def _fallback_enter_listener():
    console.print("[yellow]⚠ Fallback: nhấn Enter để kích hoạt PhysBot[/yellow]")
    while True:
        try:
            input()
            _activate_bot()
        except EOFError:
            time.sleep(3)


# ══════════════════════════════════════════════════════════════════
# STATE TRANSITIONS
# ══════════════════════════════════════════════════════════════════

def _activate_bot():
    global _bot_state, _last_active_ts
    with _state_lock:
        if _bot_state == STATE_ACTIVE:
            return
        _bot_state      = STATE_ACTIVE
        _last_active_ts = time.time()
    while not _record_queue.empty():
        try:
            _record_queue.get_nowait()
        except Empty:
            break
    console.print("[cyan]══ ACTIVE — Tui đang nghe! ══[/cyan]")
    _beep(880, 0.25)
    _activated_event.set()


def _deactivate_bot(reason: str = "timeout"):
    global _bot_state, _last_deactivate_ts
    with _state_lock:
        _bot_state = STATE_IDLE
    _last_deactivate_ts = time.time()
    console.print(f"[dim]── IDLE ({reason}) ──[/dim]")
    _beep(440, 0.5, 1.0)
    _activated_event.clear()


def _idle_timeout_watcher():
    global _last_active_ts
    while True:
        time.sleep(5)
        if _bot_state == STATE_ACTIVE:
            if _tts_playing:
                _last_active_ts = time.time()
                continue
            elapsed = time.time() - _last_active_ts
            if elapsed >= IDLE_TIMEOUT_SEC:
                console.print(f"[dim]Timeout {IDLE_TIMEOUT_SEC:.0f}s → về IDLE[/dim]")
                _deactivate_bot("timeout")


# ══════════════════════════════════════════════════════════════════
# RECORD QUESTION
# ══════════════════════════════════════════════════════════════════

def record_question(
    silence_threshold: float = ENERGY_MIN,
    silence_duration:  float = 3.0,
    max_duration:      float = 30.0,
) -> np.ndarray:
    chunk_duration = UNIFIED_CHUNK_MS / 1000.0
    silent_chunks  = 0
    started        = False
    collected      = []
    deadline       = time.time() + max_duration

    console.print(
        f"[dim]Ghi âm từ queue (ngưỡng={silence_threshold:.3f}, "
        f"dừng sau {silence_duration:.0f}s im lặng)...[/dim]"
    )

    while True:
        if time.time() >= deadline:
            console.print(f"[dim]Max {max_duration:.0f}s → tự dừng[/dim]")
            break
        if _bot_state != STATE_ACTIVE:
            break

        try:
            chunk = _record_queue.get(timeout=0.5)
        except Empty:
            if started:
                silent_chunks += 1
                if silent_chunks >= silence_duration / chunk_duration:
                    break
            continue

        collected.append(chunk)
        energy = np.abs(chunk).mean()

        if not started:
            if energy > silence_threshold:
                started = True
                console.print("[dim]Đang nghe câu hỏi...[/dim]")
            continue

        if energy < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks >= silence_duration / chunk_duration:
            break

    if not collected:
        return np.zeros(TARGET_SR, dtype=np.float32)
    return np.concatenate(collected).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════════════

def sanitize_for_tts(text: str) -> str:
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
    text = re.sub(r'(?m)^#{1,6}\s*', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'(?m)^\s*[-•]\s+', '', text)
    unit_compounds = [
        (r'\bm/s²\b', 'mét trên giây bình phương'),
        (r'\bm/s2\b', 'mét trên giây bình phương'),
        (r'\bkg/m³\b', 'kilôgam trên mét khối'),
        (r'\bkg/m3\b', 'kilôgam trên mét khối'),
        (r'\bN/m²\b', 'niutơn trên mét vuông'),
        (r'\bN/m2\b', 'niutơn trên mét vuông'),
        (r'\bW/m²\b', 'oát trên mét vuông'),
        (r'\bJ/kg\b', 'jun trên kilôgam'),
        (r'\bJ/mol\b', 'jun trên mol'),
        (r'\bN/C\b', 'niutơn trên culông'),
        (r'\bV/m\b', 'vôn trên mét'),
        (r'\bA/m\b', 'ampe trên mét'),
        (r'\bΩ\.m\b', 'ôm nhân mét'),
        (r'\bm/s\b', 'mét trên giây'),
        (r'\bcm/s\b', 'xentimét trên giây'),
        (r'\bkm/h\b', 'kilômét trên giờ'),
        (r'\brad/s\b', 'radian trên giây'),
        (r'\brad/s²\b', 'radian trên giây bình phương'),
        (r'\bN/m\b', 'niutơn trên mét'),
        (r'\bμF\b', 'micrô fara'), (r'\bμH\b', 'micrô henry'),
        (r'\bμC\b', 'micrô culông'), (r'\bμA\b', 'micrô ampe'),
        (r'\bGHz\b', 'giga héc'), (r'\bMHz\b', 'mêga héc'),
        (r'\bkHz\b', 'kilô héc'), (r'\bMΩ\b', 'mêga ôm'),
        (r'\bkΩ\b', 'kilô ôm'), (r'\bkW\b', 'kilô oát'),
        (r'\bkV\b', 'kilô vôn'), (r'\bkJ\b', 'kilô jun'),
        (r'\bkm\b', 'kilômét'), (r'\bnF\b', 'nano fara'),
        (r'\bnC\b', 'nano culông'), (r'\bnm\b', 'nano mét'),
        (r'\bpF\b', 'picô fara'), (r'\beV\b', 'êlectrôn vôn'),
        (r'\bMeV\b', 'mêga êlectrôn vôn'),
        (r'(\d)\s*mH\b', r'\1 mili henry'), (r'(\d)\s*mA\b', r'\1 mili ampe'),
        (r'(\d)\s*mV\b', r'\1 mili vôn'), (r'(\d)\s*ms\b', r'\1 mili giây'),
        (r'(\d)\s*mm\b', r'\1 milimét'),
        (r'\bm²\b', 'mét vuông'), (r'\bm2\b', 'mét vuông'),
        (r'\bcm²\b', 'xentimét vuông'), (r'\bcm2\b', 'xentimét vuông'),
        (r'\bm³\b', 'mét khối'), (r'\bm3\b', 'mét khối'),
        (r'\bcm³\b', 'xentimét khối'), (r'\bcm3\b', 'xentimét khối'),
        (r'\bs²\b', 'giây bình phương'), (r'\bs2\b', 'giây bình phương'),
        (r'\bHz\b', 'héc'), (r'\bPa\b', 'pascal'), (r'\batm\b', 'atmôtphe'),
        (r'\bWb\b', 'vêbe'), (r'\brad\b', 'radian'), (r'\bmol\b', 'mol'),
        (r'\bkg\b', 'kilôgam'), (r'\bcm\b', 'xentimét'),
        (r'\bmm\b', 'milimét'), (r'\bmin\b', 'phút'),
    ]
    for pattern, repl in unit_compounds:
        text = re.sub(pattern, repl, text)
    for sym, name in [('N','niutơn'),('J','jun'),('W','oát'),('V','vôn'),
                      ('A','ampe'),('F','fara'),('H','henry'),('T','tesla'),
                      ('C','culông'),('K','ken-vin')]:
        text = re.sub(rf'(\d)\s*{re.escape(sym)}\b', rf'\1 {name}', text)
    text = re.sub(r'(\d)\s*Ω', r'\1 ôm', text)
    text = re.sub(r'\bΩ\b', 'ôm', text)
    for pattern, repl in [
        (r'10\^-34','mười mũ trừ ba mươi bốn'),(r'10\^-31','mười mũ trừ ba mươi mốt'),
        (r'10\^-27','mười mũ trừ hai mươi bảy'),(r'10\^-23','mười mũ trừ hai mươi ba'),
        (r'10\^-19','mười mũ trừ mười chín'),(r'10\^-15','mười mũ trừ mười lăm'),
        (r'10\^-12','mười mũ trừ mười hai'),(r'10\^-9','mười mũ trừ chín'),
        (r'10\^-6','mười mũ trừ sáu'),(r'10\^-3','mười mũ trừ ba'),
        (r'10\^9','mười mũ chín'),(r'10\^8','mười mũ tám'),
        (r'10\^6','mười mũ sáu'),(r'10\^3','mười mũ ba'),(r'10\^2','mười mũ hai'),
        (r'\^2\b',' bình phương'),(r'\^3\b',' lập phương'),
        (r'\^-1\b',' mũ trừ một'),(r'\^-2\b',' mũ trừ hai'),(r'\^-3\b',' mũ trừ ba'),
    ]:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    for old, new in [('₀',' không'),('₁',' một'),('₂',' hai'),('₃',' ba'),
                     ('₄',' bốn'),('₅',' năm'),('₆',' sáu'),('₇',' bảy'),
                     ('₈',' tám'),('₉',' chín')]:
        text = text.replace(old, new)
    for old, new in [('²',' bình phương'),('³',' lập phương'),('⁻',' trừ '),
                     ('⁰',' mũ không'),('⁴',' mũ bốn'),('⁵',' mũ năm'),
                     ('⁶',' mũ sáu'),('⁷',' mũ bảy'),('⁸',' mũ tám'),
                     ('⁹',' mũ chín'),('ⁿ',' mũ n'),('¹',' mũ một')]:
        text = text.replace(old, new)
    text = re.sub(r'√\(([^)]+)\)', r'căn bậc hai của \1', text)
    text = text.replace('√', 'căn bậc hai của ')
    for old, new in [('½',' một phần hai '),('¼',' một phần bốn '),
                     ('¾',' ba phần bốn '),('⅓',' một phần ba '),('⅔',' hai phần ba ')]:
        text = text.replace(old, new)
    for old, new in [
        ('α','anpha'),('β','bê-ta'),('γ','gama'),('ω','ô-mê-ga'),
        ('λ','lăm-đa'),('θ','tê-ta'),('π','pi'),('Δ','delta '),('δ','delta '),
        ('Φ','phi '),('φ','phi '),('Σ','tổng '),('σ','sigma '),('μ','muy'),
        ('η','êta'),('ρ','rô'),('ε','êp-xi-lông'),('τ','tô'),('ξ','xi'),
        ('Λ','lăm-đa'),('Ω','ô-mê-ga hoa'),
    ]:
        text = text.replace(old, new)
    text = re.sub(r'([a-zA-ZÀ-ỹ])(\d)', r'\1 \2', text)
    text = re.sub(r'\s+x\s+', ' nhân ', text)
    for old, new in [('×',' nhân '),('÷',' chia '),('≈',' xấp xỉ '),
                     ('≠',' khác '),('≥',' lớn hơn hoặc bằng '),
                     ('≤',' nhỏ hơn hoặc bằng '),('>',' lớn hơn '),('<',' nhỏ hơn '),
                     ('→',' suy ra '),('⇒',' suy ra '),('∞','vô cực'),
                     ('%',' phần trăm'),('°',' độ'),('=',' bằng ')]:
        text = text.replace(old, new)
    text = text.replace('+', ' cộng ')
    for old, new in [('—',', '),('–',', '),('−',' trừ '),
                     ('━',''),('═',''),('│',''),('┃',''),
                     ('❌',''),('✅',''),('✔',''),('✗',''),('_',' ')]:
        text = text.replace(old, new)
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\s([.,;:])', r'\1', text)
    text = re.sub(r'([.,;:]){2,}', r'\1', text)
    return text.strip()


def _flush_record_queue():
    flushed = 0
    while not _record_queue.empty():
        try:
            _record_queue.get_nowait()
            flushed += 1
        except Empty:
            break
    if flushed:
        console.print(f"[dim]Flush {flushed} chunk echo sau TTS[/dim]")


def _post_tts_cleanup(extra_mute: float = 0.0):
    global _tts_playing, _last_active_ts, _stream_warmed_up
    total_mute = POST_TTS_MUTE_SEC + extra_mute
    time.sleep(total_mute)
    _flush_record_queue()
    if _unified_stream is not None:
        try:
            if not _unified_stream.active:
                _stream_warmed_up = False
                _unified_stream.start()
                console.print(f"[dim]✓ Mic stream restart OK[/dim]")
        except Exception as e:
            console.print(f"[dim red]Mic restart lỗi: {e}[/dim red]")
    _last_active_ts = time.time()
    _tts_playing    = False
    if _bot_state == STATE_ACTIVE:
        _beep(880, 0.15, 1.0)
    console.print(f"[dim]Mic bật lại (mute {total_mute:.1f}s)[/dim]")


def plify_mp3(src_path: str, gain_db: float = 30.0) -> str:
    """
    [v3.2.0] Khuếch đại file MP3 bằng ffmpeg (thay pydub).

    Lý do đổi: pydub cần libav/ffmpeg riêng và hay lỗi trên Pi OS Lite.
    ffmpeg đã có sẵn trong Raspberry Pi OS, nhanh hơn, không cần thư viện Python.

    gain_db=30 ≈ tăng ~300% âm lượng (pydub dùng +dB tuyến tính, ffmpeg dùng volume filter).
    Quy đổi: 30 dB → factor ~31.6×  (quá to) → dùng volume=4.0 (~12dB) cho an toàn.
    Điều chỉnh qua biến môi trường TTS_VOLUME_FACTOR (mặc định 4.0).
    """
    volume_factor = float(os.getenv("TTS_VOLUME_FACTOR", "4.0"))
    out_path = src_path.replace(".mp3", "p.mp3")
    try:
        import subprocess
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-filter:a", f"volume={volume_factor}",
                "-codec:a", "libmp3lame",
                "-q:a", "2",
                out_path,
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0 and os.path.exists(out_path):
            os.unlink(src_path)
            console.print(f"[dim]ffmpeg amplify ×{volume_factor} OK[/dim]")
            return out_path
        else:
            stderr = result.stderr.decode(errors="ignore")[-200:]
            console.print(f"[yellow]⚠ ffmpeg lỗi (rc={result.returncode}): {stderr}[/yellow]")
            return src_path
    except FileNotFoundError:
        console.print("[yellow]⚠ ffmpeg không tìm thấy — dùng file gốc (âm lượng thấp)[/yellow]")
        return src_path
    except subprocess.TimeoutExpired:
        console.print("[yellow]⚠ ffmpeg timeout — dùng file gốc[/yellow]")
        return src_path
    except Exception as e:
        console.print(f"[yellow]⚠ plify_mp3 lỗi: {e} — dùng file gốc[/yellow]")
        return src_path


async def speak(text: str):
    global _tts_playing, _last_active_ts
    text = sanitize_for_tts(text).strip()
    if not text:
        return
    if text[-1] not in '.!?':
        text += '.'
    console.print(f"[magenta]TTS ({len(text)} ký tự): {repr(text[:80])}[/magenta]")
    _tts_playing    = True
    _last_active_ts = time.time()

    if _unified_stream is not None:
        try:
            if _unified_stream.active:
                _unified_stream.stop()
        except Exception:
            pass

    _init_pygame_mixer()

    try:
        import edge_tts
        import pygame
        voice = os.getenv("TTS_VOICE", "vi-VN-HoaiMyNeural")
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        await communicate.save(tmp_path)
        tmp_path = plify_mp3(tmp_path)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            _last_active_ts = time.time()
            pygame.time.wait(100)
        time.sleep(0.3)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        os.unlink(tmp_path)
        extra_mute = min(len(text) / 200, 2.0)
        threading.Thread(target=_post_tts_cleanup, args=(extra_mute,), daemon=True).start()
        return
    except ImportError:
        pass
    except Exception as e:
        console.print(f"[yellow]edge-tts lỗi: {e}, fallback gTTS[/yellow]")

    try:
        from gtts import gTTS
        import pygame
        tts = gTTS(text=text, lang='vi', slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        tts.save(tmp_path)
        tmp_path = plify_mp3(tmp_path)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            _last_active_ts = time.time()
            pygame.time.wait(100)
        time.sleep(0.3)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        os.unlink(tmp_path)
    except Exception as e:
        console.print(f"[red]TTS lỗi hoàn toàn: {e}[/red]")
    finally:
        extra_mute = min(len(text) / 200, 2.0)
        threading.Thread(target=_post_tts_cleanup, args=(extra_mute,), daemon=True).start()


# ══════════════════════════════════════════════════════════════════
# PROCESS ONE TURN
# ══════════════════════════════════════════════════════════════════

def process_turn(text: str):
    global _last_response, _last_numbers, _last_topic, _last_active_ts
    _last_active_ts = time.time()

    corrected = correct_physics_text(text, use_llm=False)
    log_correction(text, corrected)
    if corrected != text:
        console.print(f"[yellow]Fixed: {corrected}[/yellow]")

    if _is_repeat_request(corrected) and _last_response:
        _log_implicit("repeat_request", question=corrected[:80])
        console.print("[yellow]→ Phát lại response trước[/yellow]")
        asyncio.run(speak(_last_response))
        return

    if _is_not_understand(corrected) and _last_response:
        _log_implicit("not_understand", question=corrected[:80])

    current_topic = _extract_topic(corrected)
    if _last_topic and current_topic and _last_topic == current_topic:
        _log_implicit("rephrase_same", topic=current_topic, question=corrected[:80])
    _last_topic = current_topic

    if _is_keep_data_request(corrected) and _last_numbers:
        corrected = f"{corrected}. Số liệu đã cho từ bài trước: {_last_numbers}"
        console.print(f"[dim]→ Inject số liệu cũ: {_last_numbers}[/dim]")

    extracted = _extract_numbers(corrected)
    if extracted:
        _last_numbers = extracted

    _log_implicit("question", question=corrected[:120])

    asyncio.run(speak("Oke, đợi tui một chút nhé!"))

    with console.status("Đang xử lý...", spinner="dots"):
        t0       = time.time()
        response = get_response(corrected)
        t1       = time.time()

    _last_response  = response
    _last_active_ts = time.time()

    _log_implicit("answer", question=corrected[:80], answer=response[:120],
                  api_sec=round(t1 - t0, 2))

    if len(response) > 2000:
        chunk = response[:2000]
        cut   = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
        response = response[:cut+1] + " (tui rút gọn nha)" if cut > 0 else response[:2000]

    console.print(f"[cyan]PhysBot: {response}[/cyan]")
    console.print(f"[dim]API: {t1-t0:.2f}s[/dim]")
    asyncio.run(speak(response))
    _last_active_ts = time.time()


# ══════════════════════════════════════════════════════════════════
# ACTIVE LOOP
# ══════════════════════════════════════════════════════════════════

def active_loop():
    turn_count = 0
    while True:
        if _bot_state != STATE_ACTIVE:
            break

        audio_np = record_question()

        if _bot_state != STATE_ACTIVE:
            break

        if audio_np.size == 0:
            time.sleep(0.1)
            continue

        energy = np.abs(audio_np).mean()
        if energy < ENERGY_MIN * 0.5:
            console.print(f"[dim]Energy {energy:.4f} quá thấp, bỏ qua[/dim]")
            time.sleep(0.1)
            continue

        with console.status("Nhận dạng giọng nói...", spinner="dots"):
            raw_text = transcribe(audio_np, language="vi")

        if not raw_text.strip():
            console.print("[dim]Không nhận ra giọng nói[/dim]")
            continue

        console.print(f"[yellow]Bạn: {raw_text}[/yellow]")
        turn_count += 1
        process_turn(raw_text)

    console.print(f"[dim]Active loop kết thúc sau {turn_count} lượt[/dim]")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    console.print("[cyan]╔══════════════════════════════════════╗[/cyan]")
    console.print("[cyan]║     PhysBot Pi Client v3.2.0         ║[/cyan]")
    console.print("[cyan]║   Smart Guide: HC-SR04 + OpenCV      ║[/cyan]")
    console.print("[cyan]╚══════════════════════════════════════╝[/cyan]")
    console.print(f"[cyan]Server   : {API_BASE}[/cyan]")
    console.print(f"[cyan]Session  : {SESSION_ID}[/cyan]")
    console.print(f"[cyan]Wake     : model='{WAKE_MODEL}' threshold={WAKE_THRESHOLD}[/cyan]")
    console.print(f"[cyan]Energy   : {ENERGY_MIN} | PostTTS mute: {POST_TTS_MUTE_SEC}s[/cyan]")
    console.print(f"[cyan]Timeout  : {IDLE_TIMEOUT_SEC:.0f}s → IDLE[/cyan]")
    console.print(f"[cyan]Warmup   : {STREAM_WARMUP_SEC:.0f}s sau khi stream start[/cyan]")
    console.print(f"[cyan]Mic      : device={INPUT_DEVICE} @ {HW_SR}Hz/{HW_CHANNELS}ch → {TARGET_SR}Hz[/cyan]")
    console.print(f"[cyan]Camera   : khoảng tối ưu {DIST_MIN_CM:.0f}–{DIST_MAX_CM:.0f}cm "
                   f"(ideal {DIST_IDEAL_CM:.0f}cm) | blur≥{BLUR_MIN_SCORE}[/cyan]")
    console.print(f"[cyan]Offset   : ngưỡng lệch trái/phải >{OFFSET_THRESHOLD*100:.0f}% frame[/cyan]")
    hf_status = HF_DATASET_REPO if (HF_TOKEN and HF_DATASET_REPO) else "không cấu hình"
    console.print(f"[cyan]HF Log   : {hf_status}[/cyan]")
    console.print()

    try:
        from scipy.signal import resample_poly
        console.print("[green]✓ scipy OK[/green]")
    except ImportError:
        console.print("[red]✗ scipy chưa cài! Chạy: pip install scipy[/red]")
        sys.exit(1)

    try:
        with httpx.Client(timeout=5) as client:
            r    = client.get(f"{API_BASE}/health")
            info = r.json()
            console.print(
                f"[green]Server OK — model: {info.get('model','?')} | "
                f"chromadb: {info.get('chromadb','?')}[/green]"
            )
    except Exception:
        console.print(f"[yellow]⚠ Không kết nối được server tại {API_BASE}[/yellow]")

    _init_pygame_mixer()
    model_ok = _load_wake_model()
    _unified_stream = _start_unified_stream()

    if not model_ok:
        threading.Thread(target=_fallback_enter_listener, daemon=True).start()

    if _unified_stream is None:
        console.print("[yellow]⚠ Chạy không có mic — chỉ dùng Enter mode[/yellow]")
        threading.Thread(target=_fallback_enter_listener, daemon=True).start()

    threading.Thread(target=_idle_timeout_watcher, daemon=True).start()

    wake_hint = "PhysBot" if WAKE_MODEL.endswith(".onnx") else WAKE_MODEL.replace("_", " ").title()
    console.print(f"\n[dim]IDLE — Nói '{wake_hint}' để bắt đầu![/dim]\n")

    try:
        while True:
            _activated_event.wait()
            _activated_event.clear()
            active_loop()
            console.print("[dim]IDLE — đang chờ wake word...[/dim]")
    except KeyboardInterrupt:
        duration_min = round((time.time() - SESSION_START_TIME) / 60, 1)
        _log_implicit("session_end", duration_minutes=duration_min)
        console.print(f"\n[dim]Session {SESSION_ID}: {duration_min} phút[/dim]")
        _log_to_hf()
        console.print("[red]Thoát...[/red]")
        if _unified_stream:
            _unified_stream.stop()
            _unified_stream.close()






