"""
Transcriber module using WhisperX for GPU-accelerated transcription with
VAD pre-processing and word-level alignment.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import torch
import whisperx
import yaml

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
VAD_ONSET = 0.500
VAD_OFFSET = 0.363
INITIAL_BATCH_SIZE = 16
FALLBACK_BATCH_SIZE = 8

# Repetition filter thresholds
MAX_CONSECUTIVE_REPEATS = 6
MIN_DURATION_FOR_FILTER = 8.0  # seconds


def _has_excessive_repetition(text: str) -> bool:
    """Check if any token appears 6+ times consecutively (case-insensitive)."""
    tokens = text.strip().split()
    if len(tokens) < MAX_CONSECUTIVE_REPEATS:
        return False
    count = 1
    for i in range(1, len(tokens)):
        if tokens[i].lower() == tokens[i - 1].lower():
            count += 1
            if count >= MAX_CONSECUTIVE_REPEATS:
                return True
        else:
            count = 1
    return False


def transcribe(video_path: str) -> dict:
    """Transcribe a video file using WhisperX with word-level alignment.

    Args:
        video_path: Path to the video/audio file to transcribe.

    Returns:
        A dict with ``"backend"``, ``"language"``, and ``"segments"`` keys.
        Each segment contains ``"start"``, ``"end"``, ``"text"``, and
        optionally ``"words"`` (if alignment succeeded).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")

    # ── Load config ──────────────────────────────────────────────
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_name = config.get("whisper_model", "turbo")

    # ── Cache path ───────────────────────────────────────────────
    output_dir = Path("output/transcriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{video_path.stem}.json"

    # ── Return cached result if available ────────────────────────
    if json_path.exists():
        logger.info("Transcrição em cache: %s", json_path)
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    # ── Load audio ───────────────────────────────────────────────
    logger.info("Carregando áudio: %s", video_path.name)
    audio = whisperx.load_audio(str(video_path))

    # ── Load WhisperX model ──────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s | Modelo WhisperX: %s", device.upper(), model_name)
    asr_options = {
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "log_prob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
    }
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type="int8_float16",
        asr_options=asr_options,
    )
    logger.info(f"Modelo carregado — device: {model.device}")

    # ── Multi-point language detection ────────────────────────────
    SAMPLE_RATE = 16000
    WINDOW_SECONDS = 30
    WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE

    total_samples = audio.shape[0]
    total_duration = total_samples / SAMPLE_RATE

    # Window start positions (seconds)
    window_starts = {
        "A": 60,
        "B": total_duration * 0.20,
        "C": total_duration * 0.40,
    }

    detected = {}
    for label, start_sec in window_starts.items():
        start_sample = int(start_sec * SAMPLE_RATE)
        if start_sample + WINDOW_SAMPLES > total_samples:
            logger.info("Janela %s: pulando (áudio curto demais)", label)
            continue
        chunk = audio[start_sample : start_sample + WINDOW_SAMPLES]
        result = model.model.detect_language(chunk)
        lang = result[0]
        detected[label] = lang
        logger.info(
            "Janela %s (início=%.1fs): idioma detectado=%s",
            label,
            start_sec,
            lang,
        )

    if not detected:
        language = "en"
    else:
        counts = Counter(detected.values())
        most_common = counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            language = detected.get("B", most_common[0][0])
            logger.info("Empate — usando janela B: %s", language)
        else:
            language = most_common[0][0]

    logger.info("Idioma final detectado: %s", language)

    # ── Transcribe ────────────────────────────────────────────────
    batch_size = INITIAL_BATCH_SIZE
    while True:
        try:
            result = model.transcribe(
                audio,
                batch_size=batch_size,
                language=language,
            )
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size > FALLBACK_BATCH_SIZE:
                logger.warning(
                    "OOM com batch_size=%d, tentando batch_size=%d",
                    batch_size,
                    FALLBACK_BATCH_SIZE,
                )
                batch_size = FALLBACK_BATCH_SIZE
                torch.cuda.empty_cache()
                continue
            raise

    audio_duration = total_duration
    n_segments = len(result["segments"])
    logger.info(
        "Transcrição concluída — idioma=%s, duração=%.1fs, segmentos=%d",
        language,
        audio_duration,
        n_segments,
    )

    # ── Filter excessive repetition ──────────────────────────────
    raw_segments = result["segments"]
    filtered_segments: list[dict] = []
    dropped_count = 0
    for seg in raw_segments:
        duration = seg.get("end", 0.0) - seg.get("start", 0.0)
        text = seg.get("text", "").strip()
        if duration > MIN_DURATION_FOR_FILTER and _has_excessive_repetition(text):
            logger.warning(
                "Segmento dropado (repetição >%ds): %s", MIN_DURATION_FOR_FILTER, text
            )
            dropped_count += 1
            continue
        filtered_segments.append(seg)

    logger.info(
        "Filtro: %d segmentos mantidos, %d dropados",
        len(filtered_segments),
        dropped_count,
    )

    # ── Word-level alignment ─────────────────────────────────────
    alignment_ran = False
    aligned_segments: list[dict] = []

    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
        aligned_result = whisperx.align(
            filtered_segments,
            align_model,
            align_metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )
        alignment_ran = True
        aligned_segments = aligned_result.get("segments", [])
        logger.info("Alinhamento palavra-a-palavra concluído")
    except Exception as exc:
        logger.warning(
            "Alinhamento não disponível para idioma '%s': %s. "
            "Salvando sem campo 'words'.",
            language,
            exc,
        )
        # Fall back — build segments with only start / end / text
        aligned_segments = [
            {
                "start": s.get("start", 0.0),
                "end": s.get("end", 0.0),
                "text": s.get("text", "").strip(),
            }
            for s in filtered_segments
        ]

    # ── Build output ─────────────────────────────────────────────
    output_segments: list[dict] = []
    for seg in aligned_segments:
        entry: dict = {
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": seg.get("text", "").strip(),
        }
        if alignment_ran and "words" in seg:
            words = seg["words"]
            entry["words"] = [
                {
                    "word": w.get("word", ""),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "score": float(w.get("score", 0.0)),
                }
                for w in words
                if w.get("word", "").strip()
            ]
        output_segments.append(entry)

    output = {
        "backend": "whisperx",
        "language": language,
        "segments": output_segments,
    }

    # ── Save JSON ────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Transcrição salva em: %s", json_path)
    return output
