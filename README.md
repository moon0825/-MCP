# meeting-scribe

회의 녹음 파일 → **화자를 학습하는 전사** → 프로젝트 컨텍스트 기반 **회의록 자동 작성**을 위한 MCP 서버.

- 🎙️ 폰 녹음(m4a 등) 어떤 포맷이든 처리 (ffmpeg 내장 — 별도 설치 불필요)
- 👥 화자분리는 **로컬 CPU**(sherpa-onnx, ~35MB 모델) — 목소리 데이터가 밖으로 안 나감
- 🧠 화자 학습: 한 번 "이 사람이 김대리"라고 확정하면 **다음 회의부터 자동 인식**
- 📚 프로젝트 용어집·참석자·이전 회의 요약이 쌓이며 전사 정확도와 회의록 품질이 갈수록 향상
- 🔌 ASR 엔진 교체 가능: Groq(무료) / Gemini(무료) / ReturnZero(한국어 특화) / 로컬 Whisper(완전 오프라인)
- 💻 Zoom 회의는 참가자별 오디오 트랙으로 받으면 화자 정확도 100% + 화자 학습 부트스트랩

설계 배경과 기술 선택 근거는 [DESIGN.md](DESIGN.md) 참고.

## 설치 (Windows/macOS/Linux)

```bash
# uv 권장 (https://docs.astral.sh/uv/)
git clone https://github.com/moon0825/-MCP meeting-scribe && cd meeting-scribe
uv venv && uv pip install -e .          # 기본 (클라우드 ASR)
uv pip install -e ".[local]"            # 선택: 완전 로컬 whisper (민감 회의용)
```

### Claude Code에 등록

```bash
claude mcp add meeting-scribe --env GROQ_API_KEY=<키> -- uv --directory <설치경로> run meeting-scribe
```

환경변수 (하나 이상 설정, 없으면 `[local]` 설치 필요):

| 변수 | 엔진 | 비고 |
|---|---|---|
| `GROQ_API_KEY` | Groq Whisper large-v3-turbo | 무료 티어, 기본 추천 |
| `GEMINI_API_KEY` | Gemini 2.5 Flash | 무료 티어. 단어 타임스탬프가 없어 화자 배정 정밀도 낮음 |
| `RTZR_CLIENT_ID` + `RTZR_CLIENT_SECRET` | ReturnZero | 한국어 최고 정확도 + 키워드 부스팅. 무료 10시간 |
| `MEETING_SCRIBE_ASR` | 기본 엔진 강제 (`rtzr`/`groq`/`gemini`/`local`) | |

## 사용 흐름 (Claude Code에서)

```
1. "무인공장 프로젝트 만들어줘. 참석자는 김대리, 박과장. 용어는 AMR, MFM 등록"
   → create_project / add_participants / add_glossary
2. "이 녹음 전사해줘: C:\Users\me\Recordings\회의.m4a"
   → submit_transcription → (수 분 뒤) get_transcript
3. "화자2가 김대리야" (본인 동의 확인 후)
   → confirm_speaker → 다음 회의부터 자동 제안
4. "회의록 작성해서 저장해줘"
   → Claude가 프로젝트 컨텍스트로 [?] 구간을 추론·보정하며 회의록 작성 → save_minutes
   → 요약·신규 용어가 축적되어 다음 회의가 더 정확해짐
```

## 프라이버시 / 법적 주의

- 화자 보이스프린트(음성 임베딩)는 **개인정보보호법상 민감정보(생체인식정보)** 입니다.
  `confirm_speaker`는 당사자 동의 확인(`consent_confirmed=True`) 없이는 등록을 거부하며,
  `purge_speaker`로 언제든 완전 삭제할 수 있습니다. 임베딩은 로컬 SQLite에만 저장됩니다.
- 민감한 회의는 `asr="local"`로 완전 오프라인 처리하세요.
  - NVIDIA GPU 자동 감지: RTX 3050급(VRAM 4GB+)이면 `large-v3-turbo`가 실시간보다 빠르게 동작,
    GTX 1050(2GB)은 `small`로 자동 축소. CUDA 런타임이 없으면 CPU int8 폴백 (오디오의 1~3배 시간).
  - `system_check` 툴이 현재 머신의 실행 계획을 알려줍니다.
- 데이터 위치: `~/.meeting-scribe/` (`MEETING_SCRIBE_DATA`로 변경 가능). 리포에 커밋 금지.

## 개발

```bash
uv pip install -e ".[local]"
python scripts/smoke_test.py   # 화자분리→전사→학습→재인식 엔드투엔드 테스트
```
