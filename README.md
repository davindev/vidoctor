# Vidoctor

AI 영상 감수 에이전트 — 영상을 업로드하면 5차원으로 분석하고 개선점을 제안합니다.

## 5차원 분석

| 차원 | 방법 | 활성 카테고리 |
|---|---|---|
| Filler | WhisperX(faster-whisper-large-v3-turbo + wav2vec2 정렬) + 한국어 사전·정규식 | 전체 |
| 말 속도 (CPS) | Net CPS 슬라이딩 윈도우 (5s/1s), 절대 기준 AND ±2σ | 전체 |
| 시각 dead zone | OpenCV diff + SSIM + ASR 무발화, 카테고리별 임계값 | 전체 |
| 시선 이탈 | MediaPipe Face Mesh iris + cv2.solvePnP head pose | 강의 |
| 내용 공백 | GPT-4o Vision multi-image batch + ASR 동시 input + rubric | 강의·기타 |

카테고리: **강의 / 브이로그·인터뷰 / 기타** (사용자 드롭다운 선택)

## 기술 스택

- 오케스트레이션: LangGraph + LangChain
- VLM: GPT-4o Vision (sync) + GPT-4o-mini (개선 제안)
- 인프라: Supabase Postgres + Storage
- UI: Streamlit
- 관찰성: Langfuse (LLM trace) + MLflow (실험 추적)
- 평가: scikit-learn + Cohen's κ + DeepEval + Label Studio + pytest
- 배포: Fly.io performance-2x + Docker

## 처리 목표 (3분 영상)

- 분석 시간: ≤ 1.5분
- 비용: ≤ $0.20/영상

## 셋업

```bash
# Python + uv (mise가 자동 활성화)
mise install

# 의존성 설치
uv sync

# 환경 변수
cp .env.example .env
# .env 채우기

# Streamlit 실행
uv run streamlit run src/vidoctor/ui/app.py

# 테스트
uv run pytest
```

## 라이선스

MIT (예정)
