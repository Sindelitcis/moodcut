"""Analyze transcript chunks via DeepSeek API to find highlight moments."""

import json
import os
import re
import sys
from pathlib import Path

import httpx
import yaml

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
REQUEST_TIMEOUT = 120


def analyze(chunks: list[str], metadata: dict) -> list[dict]:
    """Send each transcript chunk to DeepSeek and collect highlight moments.

    Args:
        chunks: List of formatted transcript chunks (each a single string).
        metadata: Dict with ``"title"``, ``"channel"`` and ``"url"`` keys.

    Returns:
        List of approved highlight segments sorted by confidence.
    """
    # ── Load config ──────────────────────────────────────────────
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    threshold = float(config.get("confidence_threshold", 0.75))

    # ── Cache path ───────────────────────────────────────────────
    safe_name = _sanitize_title(metadata.get("title", "analysis"))
    output_dir = Path("output/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{safe_name}.json"

    # ── Return cached result if available ────────────────────────
    if json_path.exists():
        print(
            f"[*] Análise em cache: {json_path}",
            file=sys.stderr,
            flush=True,
        )
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    # ── API key ──────────────────────────────────────────────────
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY não encontrada no ambiente. "
            "Adicione DEEPSEEK_API_KEY=... no arquivo .env"
        )

    system_prompt = (
        "Você é um especialista em análise de conteúdo para criação de clipes virais do YouTube. "
        "Sua tarefa é analisar transcrições de lives e identificar momentos que valem um corte. "
        "Retorne APENAS um array JSON válido, sem markdown, sem blocos de código, sem texto adicional. "
        "Cada objeto do array deve ter exatamente estes campos:\n"
        "- inicio: timestamp no formato HH:MM:SS\n"
        "- fim: timestamp no formato HH:MM:SS\n"
        "- humor: exatamente um de: engraçado, triste, raiva, inspirador\n"
        "- confianca: número entre 0 e 1 indicando certeza que vale um clipe\n"
        "- motivo: frase curta explicando por que esse momento vale um clipe\n"
        "- titulo_sugerido: título chamativo para o clipe no estilo YouTube\n"
        "Retorne array vazio [] se não houver momentos relevantes no chunk. "
        "Seja criterioso: prefira qualidade a quantidade. Só inclua momentos realmente marcantes."
    )

    all_results: list[dict] = []

    # ── Process each chunk ───────────────────────────────────────
    for i, chunk in enumerate(chunks, 1):
        print(
            f"[*] Analisando chunk {i}/{len(chunks)}...",
            file=sys.stderr,
            flush=True,
        )

        user_prompt = (
            f"Canal: {metadata.get('channel', '')}\n"
            f"Título: {metadata.get('title', '')}\n\n"
            f"Transcrição (chunk {i + 1} de {len(chunks)}):\n{chunk}"
        )

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            raw_content = data["choices"][0]["message"]["content"].strip()

            # ── Parse JSON from response ─────────────────────────
            try:
                segments = json.loads(raw_content)
            except json.JSONDecodeError as e:
                print(
                    f"  [ERRO] JSON inválido no chunk {i}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"  Resposta (início): {raw_content[:300]}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            # Normalise to list
            if isinstance(segments, dict):
                segments = [segments]
            elif not isinstance(segments, list):
                print(
                    f"  [ERRO] Tipo inesperado no chunk {i}: {type(segments)}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            # ── Validate each segment ────────────────────────────
            valid: list[dict] = []
            for seg in segments:
                humor = seg.get("humor", "")
                if humor not in ("engraçado", "triste", "raiva", "inspirador"):
                    print(
                        f"  [AVISO] Humor inválido ignorado: {humor}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                seg["confianca"] = float(seg.get("confianca", 0))
                valid.append(seg)

            all_results.extend(valid)
            print(
                f"  -> {len(valid)} momento(s) válido(s)",
                file=sys.stderr,
                flush=True,
            )

        except httpx.HTTPStatusError as e:
            print(
                f"  [ERRO] HTTP {e.response.status_code} no chunk {i}",
                file=sys.stderr,
                flush=True,
            )
        except httpx.RequestError as e:
            print(
                f"  [ERRO] Falha de conexão no chunk {i}: {e}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:
            print(
                f"  [ERRO] Inesperado no chunk {i}: {e}",
                file=sys.stderr,
                flush=True,
            )

    # ── Filter by confidence threshold ───────────────────────────
    before = len(all_results)
    all_results = [seg for seg in all_results if seg["confianca"] >= threshold]
    filtered = before - len(all_results)
    if filtered:
        print(
            f"[*] {filtered} segmento(s) removido(s) por confiança < {threshold}",
            file=sys.stderr,
            flush=True,
        )

    # ── Save ─────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] Análise salva em: {json_path}",
        file=sys.stderr,
        flush=True,
    )
    return all_results


def _sanitize_title(title: str) -> str:
    """Produce a safe filesystem name from a video title."""
    safe = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip()
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe[:100] or "analysis"
