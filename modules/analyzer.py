"""Analyze transcript via DeepSeek in two passes: topic mapping then enrichment."""

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

HUMOR_VALUES = ("engraçado", "triste", "raiva", "inspirador")


def analyze(chunks: list[str], metadata: dict) -> list[dict]:
    """Two-pass analysis: topic segmentation → per-topic enrichment.

    Pass 1 — sends the full transcript and asks DeepSeek to identify natural
    topic boundaries (preferring 8–20 min segments, never cutting a topic short).

    Pass 2 — for each identified topic, sends the corresponding excerpt and
    asks for a YouTube-style title, predominant mood and short description.

    Args:
        chunks: List of formatted transcript chunk strings.
        metadata: Dict with ``"title"``, ``"channel"`` and ``"url"`` keys.

    Returns:
        List of enriched topic dicts.
    """
    # ── Load config ──────────────────────────────────────────────
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    min_dur = int(config.get("min_clip_duration", 480))
    max_dur = int(config.get("max_clip_duration", 0))

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

    # ── Build full transcript ────────────────────────────────────
    full_transcript = "\n\n".join(chunks)

    # ══════════════════════════════════════════════════════════════
    #  PASS 1 — Topic mapping
    # ══════════════════════════════════════════════════════════════
    print("[*] Passagem 1 — Mapeando tópicos...", file=sys.stderr, flush=True)

    dur_instruction = (
        f"Prefira tópicos entre {min_dur // 60} e {max_dur // 60} minutos de duração."
        if max_dur > 0
        else f"Prefira tópicos de pelo menos {min_dur // 60} minutos."
    )

    pass1_system = (
        "Você é um analista de conteúdo especializado em segmentar transcrições "
        "de vídeos longos (lives, streams, podcasts) por tópicos de conversa.\n\n"
        "Analise a transcrição completa abaixo e identifique onde cada assunto "
        "começa e termina naturalmente.\n"
        f"{dur_instruction}\n"
        "Nunca corte um assunto no meio — se um tópico durar 25 minutos, "
        "o segmento deve ter 25 minutos.\n"
        "Desconsidere (não inclua) abertura, encerramento e momentos de "
        "silêncio prolongado ou pausa técnica.\n"
        "Ignore e não inclua segmentos de propaganda, divulgação de parceiros, "
        "patrocínios ou chamadas para redes sociais.\n\n"
        "Os timestamps devem estar no formato HH:MM:SS onde HH são horas, "
        "MM são minutos e SS são segundos. Exemplo: 26 minutos = 00:26:00, "
        "não 26:00:00.\n\n"
        "Retorne APENAS um JSON array válido, sem markdown, sem blocos de "
        "código, sem texto adicional. Cada objeto:\n"
        '  {"inicio": "HH:MM:SS", "fim": "HH:MM:SS", '
        '"topico": "descrição breve do assunto"}'
    )

    pass1_payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": pass1_system},
            {
                "role": "user",
                "content": (
                    f"Transcrição completa do vídeo "
                    f'"{metadata.get("title", "")}" '
                    f'do canal "{metadata.get("channel", "")}":\n\n'
                    f"{full_transcript}"
                ),
            },
        ],
        "temperature": 0.3,
    }

    raw = _call_api(pass1_payload, api_key)
    topics: list[dict] = []
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list):
                for t in parsed:
                    if t.get("inicio") and t.get("fim") and t.get("topico"):
                        # Normalise to HH:MM:SS before storing
                        t["inicio"] = _normalize_timestamp(t["inicio"])
                        t["fim"] = _normalize_timestamp(t["fim"])
                        topics.append(t)
                print(
                    f"  -> {len(topics)} tópico(s) identificado(s)",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"  [ERRO] Tipo inesperado na passagem 1: {type(parsed)}",
                    file=sys.stderr,
                    flush=True,
                )
        except json.JSONDecodeError as e:
            print(
                f"  [ERRO] JSON inválido na passagem 1: {e}",
                file=sys.stderr,
                flush=True,
            )
            print(f"  Resposta: {raw[:300]}", file=sys.stderr, flush=True)
    else:
        print(
            "  [ERRO] Falha na chamada da passagem 1 (sem resposta)",
            file=sys.stderr,
            flush=True,
        )

    if not topics:
        print(
            "[*] Nenhum tópico identificado — retornando lista vazia",
            file=sys.stderr,
            flush=True,
        )
        # Save empty result so it's cached
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

    # ══════════════════════════════════════════════════════════════
    #  PASS 2 — Enrichment
    # ══════════════════════════════════════════════════════════════
    print(
        f"[*] Passagem 2 — Enriquecendo {len(topics)} tópico(s)...",
        file=sys.stderr,
        flush=True,
    )

    pass2_system = (
        "Você é um especialista em criar títulos chamativos para clipes do YouTube. "
        "Com base no excerto de transcrição e no nome do tópico fornecidos, "
        "retorne APENAS um JSON objeto válido, sem markdown, sem blocos de código, "
        "sem texto adicional, com estes campos.\n"
        "IMPORTANTE: todos os campos de texto devem ser escritos no mesmo idioma "
        "da transcrição fornecida (se a transcrição for em português, escreva em "
        "português; se for em inglês, escreva em inglês).\n"
        '  "titulo_sugerido": título chamativo estilo YouTube, máximo 70 caracteres. '
        "Provoca curiosidade pelo tema, nunca pelo tom. "
        "Nunca mencione humor, emoção ou tom no título. "
        "Exemplos ruins: 'com muito humor', 'de forma emocionante', 'discussão séria'.\n"
        '  "humor": exatamente um de: engraçado, triste, raiva, inspirador\n'
        '  "descricao": frase curta descrevendo o que acontece no trecho\n'
        '  "descricao_youtube": string com descrição otimizada para YouTube '
        "seguindo estas regras:\n"
        "  - Primeiras 2 linhas: gancho direto sobre o conteúdo do clipe, "
        "sem spoiler, que desperte curiosidade (essas linhas aparecem "
        "antes do 'mostrar mais')\n"
        "  - 1 linha em branco\n"
        "  - 3 a 5 hashtags específicas ao tema do clipe, separadas por espaço. "
        "Use o nome do convidado, assunto exato e nicho — evite tags genéricas "
        "como #podcast ou #humor que não ajudam no algoritmo. "
        "Lowercase, sem acento.\n"
        "  - Nunca mencionar o canal, o apresentador pelo nome, ou fazer "
        "chamadas genéricas como 'inscreva-se'\n"
        "  - Máximo 500 caracteres no total\n"
        '  "texto_thumbnail": string de 2 a 4 palavras em CAIXA ALTA — '
        "manchete de capa de revista, impacto máximo. Artigos (O, A, OS, AS) "
        "e pronomes (ELE, ELA, EU) são permitidos e bem-vindos quando completam "
        "o sentido — ex: 'SALVOU A MÃE', 'ELE CONFESSOU', 'FUI DEMITIDO', "
        "'QUASE MORRI', 'PERDEU TUDO'. Jamais verbalizar tom ou humor. "
        "IMPORTANTE: a palavra_destaque deve ser SEMPRE a última palavra "
        "(ou últimas palavras) do texto_thumbnail — nunca no meio ou no início.\n"
        '  "palavra_destaque": a 1 palavra final mais impactante do '
        "texto_thumbnail. Deve ser exatamente igual à(s) última(s) palavra(s) "
        "de texto_thumbnail. Exemplo: se texto_thumbnail='QUASE MORRI', "
        "palavra_destaque='MORRI'. Se texto_thumbnail='20 ANOS DOR', "
        "palavra_destaque='DOR'."
    )

    enriched: list[dict] = []
    total = len(topics)

    for i, topic in enumerate(topics, 1):
        print(
            f"  [tópico {i}/{total}] {topic.get('topico', '?')} "
            f"({topic['inicio']} - {topic['fim']})",
            file=sys.stderr,
            flush=True,
        )

        excerpt = _extract_excerpt(chunks, topic["inicio"], topic["fim"])
        if not excerpt:
            print(
                "    [AVISO] Nenhum trecho extraído — pulando",
                file=sys.stderr,
                flush=True,
            )
            enriched.append(topic)
            continue

        pass2_payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": pass2_system},
                {
                    "role": "user",
                    "content": (
                        f"Tópico identificado: {topic['topico']}\n"
                        f"Período: {topic['inicio']} - {topic['fim']}\n\n"
                        f"Excerto da transcrição:\n{excerpt}"
                    ),
                },
            ],
            "temperature": 0.3,
        }

        raw2 = _call_api(pass2_payload, api_key)
        if not raw2:
            print(
                "    [AVISO] Falha na chamada — usando dados básicos",
                file=sys.stderr,
                flush=True,
            )
            enriched.append(topic)
            continue

        try:
            enrichment = json.loads(raw2)
        except json.JSONDecodeError as e:
            print(
                f"    [ERRO] JSON inválido: {e} — usando dados básicos",
                file=sys.stderr,
                flush=True,
            )
            enriched.append(topic)
            continue

        # Merge with topic data
        entry = dict(topic)
        entry["titulo_sugerido"] = enrichment.get("titulo_sugerido", topic["topico"])
        mood = enrichment.get("humor", "")
        entry["humor"] = mood if mood in HUMOR_VALUES else "inspirador"
        entry["descricao"] = enrichment.get("descricao", "")
        entry["descricao_youtube"] = enrichment.get("descricao_youtube", "")
        entry["texto_thumbnail"] = enrichment.get("texto_thumbnail", "")
        entry["palavra_destaque"] = enrichment.get("palavra_destaque", "")

        enriched.append(entry)
        print(
            f"    -> humor: {entry['humor']} | título: {entry.get('titulo_sugerido', '?')[:60]}",
            file=sys.stderr,
            flush=True,
        )

    # ── Save ─────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] Análise salva em: {json_path}",
        file=sys.stderr,
        flush=True,
    )
    return enriched


# ── Helpers ──────────────────────────────────────────────────────────


def _sanitize_title(title: str) -> str:
    """Produce a safe filesystem name from a video title."""
    safe = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip()
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe[:100] or "analysis"


def _normalize_timestamp(hms: str) -> str:
    """Convert any timestamp variant to ``HH:MM:SS`` with leading zeros.

    Handles ``H:MM:SS``, ``MM:SS``, ``M:SS`` — all normalised to
    ``HH:MM:SS`` with proper carry-over (e.g. 90 minutes becomes
    ``01:30:00``, not ``00:90:00``).
    """
    parts = hms.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        m, s = int(parts[0]), int(parts[1])
        h = 0
    else:
        s = int(parts[0])
        h = m = 0

    # Carry over overflow
    carry_m, s = divmod(s, 60)
    m += carry_m
    carry_h, m = divmod(m, 60)
    h += carry_h

    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_timestamp(hms: str) -> int:
    """Convert ``HH:MM:SS`` to total seconds."""
    parts = hms.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0])


def _extract_excerpt(chunks: list[str], inicio: str, fim: str) -> str:
    """Return transcript lines whose timestamps fall within [inicio, fim].

    Both boundaries are normalised to ``HH:MM:SS`` before comparing so that
    formats like ``26:00`` or ``1:30:00`` are handled correctly.
    """
    start_sec = _parse_timestamp(_normalize_timestamp(inicio))
    end_sec = _parse_timestamp(_normalize_timestamp(fim))
    matched: list[str] = []
    for chunk in chunks:
        for line in chunk.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") and "] " in line:
                ts_part = line[1:].split("]")[0]
                try:
                    ts = _parse_timestamp(_normalize_timestamp(ts_part))
                except (ValueError, IndexError):
                    continue
                if start_sec <= ts <= end_sec:
                    matched.append(line)
    return "\n".join(matched)


def _call_api(payload: dict, api_key: str) -> str | None:
    """POST to DeepSeek and return the response text, or ``None`` on failure."""
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
        return data["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        print(
            f"  [ERRO] HTTP {e.response.status_code}",
            file=sys.stderr,
            flush=True,
        )
    except httpx.RequestError as e:
        print(
            f"  [ERRO] Falha de conexão: {e}",
            file=sys.stderr,
            flush=True,
        )
    except (KeyError, IndexError, TypeError) as e:
        print(
            f"  [ERRO] Resposta inesperada: {e}",
            file=sys.stderr,
            flush=True,
        )
    return None
