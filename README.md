# Moodcut

Pipeline automático que transforma VODs longos (lives, podcasts) em clipes prontos para o YouTube. Ele baixa o vídeo, transcreve, segmenta por tópico de conversa, corta com legenda queimada, gera thumbnail e faz upload, tudo de ponta a ponta.

**Foco atual:** conteúdo falado (just talking / podcast). Gameplay está fora de escopo (ver `STATUS.md`).

---

## Como funciona

O pipeline roda em sequência, com cache em cada etapa (etapas já feitas são puladas):

1. **Download** (`yt-dlp`): baixa o melhor MP4 do VOD (Twitch ou YouTube) em `output/downloads/`.
2. **Transcrição** (Whisper + CUDA): gera segmentos `{start, end, text}` em `output/transcriptions/`. Modelo definido no `config.yaml`.
3. **Chunking**: agrupa os segmentos em blocos de ~3 min, no formato `[HH:MM:SS] texto`, para enviar ao modelo de análise.
4. **Análise** (DeepSeek v4-flash, duas passagens):
   - Passagem 1: mapeia os tópicos de conversa (início/fim/assunto), ignorando abertura, encerramento, silêncio, propaganda e patrocínio.
   - Passagem 2: para cada tópico, gera título estilo YouTube, humor (engraçado/triste/raiva/inspirador), descrição e descrição otimizada (`descricao_youtube`).
   - Resultado salvo em `output/analysis/`.
5. **Corte + legenda** (`ffmpeg`): corta cada tópico com padding, gera o SRT a partir da transcrição e queima a legenda (re-encode via NVENC). Clipes salvos em `output/{humor}/`.
6. **Thumbnail** (`Pillow`): extrai um frame limpo do VOD original (sem legenda), escolhe o de maior variância e sobrepõe o título. Salva ao lado do clipe.
7. **Upload** (YouTube Data API v3): sobe o clipe como `unlisted`, define a thumbnail e usa a descrição otimizada.

---

## Requisitos

- **Python 3.12** (no Windows, rodar sempre com `py -3.12`, porque o 3.14 ainda não tem suporte CUDA no torch).
- **GPU NVIDIA com CUDA** (testado em RTX 4060 8GB).
- **ffmpeg + ffprobe** no PATH, com suporte a **NVENC** (`h264_nvenc`).
- **Chave da API do DeepSeek**.
- **Projeto OAuth do YouTube** (`client_secrets.json`) para o upload.

---

## Instalação

```bash
git clone https://github.com/Sindelitcis/moodcut.git
cd moodcut

py -3.12 -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

Instale o PyTorch com CUDA seguindo a recomendação oficial em https://pytorch.org (escolha a build CUDA correta para o seu sistema), e depois as demais dependências:

```bash
pip install openai-whisper yt-dlp pyyaml httpx pillow python-dotenv \
            google-auth google-auth-oauthlib google-api-python-client
```

Garanta que `ffmpeg` e `ffprobe` estão instalados e acessíveis no PATH (com NVENC).

---

## Configuração

### `config.yaml` (raiz do projeto)

```yaml
whisper_model: turbo      # modelo do Whisper (o código tem default large-v3 se ausente)
min_clip_duration: 480    # duração mínima do clipe em segundos (8 min, para monetização)
max_clip_duration: 0      # duração máxima em segundos; 0 = sem teto
padding_start: 10         # segundos adicionados antes do início do tópico
padding_end: 8            # segundos adicionados depois do fim do tópico
```

> Nota: com `max_clip_duration: 0` não há limite superior: um tópico longo vira um clipe longo. Um teto de segurança está no roadmap (ver `STATUS.md`).

### `.env` (raiz do projeto)

```
DEEPSEEK_API_KEY=sua_chave_aqui
```

### Credenciais do YouTube

- `client_secrets.json` na raiz do projeto (gerado no Google Cloud Console). **Não** vai para o Git.
- O token OAuth é salvo automaticamente em `auth/youtube_token.json` no primeiro uso (a pasta `auth/` é ignorada pelo Git, exceto o próprio `.gitignore`).

### Fontes da thumbnail

A v1 procura fontes arredondadas (Nunito-Bold, Baloo2-Bold, Comfortaa-Bold) e cai em DejaVuSans se não achar. Instale as desejadas em `C:/Windows/Fonts/` (a direção da v2 muda para fontes condensadas pesadas; ver `STATUS.md`).

---

## Uso

Rodar o pipeline completo em um VOD:

```bash
py -3.12 main.py "https://www.twitch.tv/videos/XXXXXXXXX"
```

Flags para rodar etapas isoladas (a partir do cache):

| Flag | Efeito |
|------|--------|
| `--only-thumb`  | Gera apenas as thumbnails |
| `--only-upload` | Faz apenas o upload no YouTube |
| `--skip-thumb`  | Pula a geração de thumbnails |
| `--skip-upload` | Pula o upload no YouTube |

---

## Estrutura do projeto

```
moodcut/
├── main.py              # orquestrador do pipeline
├── modules/
│   ├── downloader.py    # download do VOD (yt-dlp)
│   ├── transcriber.py   # transcrição (Whisper + CUDA)
│   ├── chunker.py       # agrupa segmentos em blocos de ~3 min
│   ├── analyzer.py      # análise em 2 passagens (DeepSeek)
│   ├── clipper.py       # corte + legenda queimada (ffmpeg/NVENC)
│   ├── thumbnailer.py   # geração de thumbnail (Pillow)
│   └── uploader.py      # upload no YouTube (Data API v3)
├── config.yaml          # parâmetros (não versionado)
├── .env                 # DEEPSEEK_API_KEY (não versionado)
├── client_secrets.json  # OAuth YouTube (não versionado)
├── auth/                # token OAuth salvo (não versionado)
├── STATUS.md            # status detalhado, decisões e roadmap
└── README.md
```

### Saída

```
output/
├── downloads/        # VOD baixado
├── transcriptions/   # JSON da transcrição
├── analysis/         # JSON da análise (tópicos enriquecidos)
└── {humor}/          # clipes (NN_titulo.mp4) + thumbnails (.jpg), por humor
```

---

## Limitações conhecidas

- **Legenda ~1s adiantada:** o corte usa stream copy, que cai no keyframe anterior ao timestamp pedido; a legenda fica timada pelo timestamp original. Correção planejada (passe único re-encodado).
- **Transcrição alucina/repete em silêncio:** modo de falha do Whisper. Correção planejada (VAD + ajustes de decodificação).
- **Sem teto de duração:** `max_clip_duration: 0` pode gerar clipes muito longos.

Detalhes, decisões e o plano de evolução (overhaul de thumbnail, teste A/B, transcrição com VAD) estão em `STATUS.md`.

---

## Licença

A definir.
