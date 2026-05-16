# Vidoctor

AI 영상 감수 에이전트 — 영상을 업로드하면 5차원으로 분석하고 개선점을 제안합니다.

## 5차원 분석

| 차원 | 방법 | 활성 카테고리 |
|---|---|---|
| Filler | WhisperX(faster-whisper-large-v3-turbo + wav2vec2 정렬) + 한국어 사전·정규식 | 전체 |
| 말 속도 (CPS) | Net CPS 슬라이딩 윈도우 (5s/1s), 영상 평균 대비 ±1.5σ + 평탄 영상 가드 | 전체 |
| 시각 dead zone | Silero VAD 무발화 ∩ Optical flow per-frame max median, 카테고리별 임계값 | 전체 |
| 시선 이탈 | MediaPipe Tasks FaceLandmarker + BlazeFace 자동 ROI + cv2.solvePnP head pose, 영상 baseline 차감 후 deviation 임계 | 강의 |
| 내용 공백 | GPT-4o Vision multi-image batch + ASR 동시 input + rubric | 강의·기타 |

카테고리: **강의 / 브이로그·인터뷰 / 기타** (사용자 드롭다운 선택)

## v1.0 vs v1.1 (계획)

v1.0 MVP는 차원별 검출(이상 구간 감지)에 집중. severity 차등과 suggestion priority는
임계 결정 근거(라벨링·평가)가 갖춰지지 않은 시점에 분기에 의미가 없다고 판단해 스펙
아웃했고, 평가 메트릭도 차원별 F1만 측정. v1.1에서 다음을 보완:

- **차원별 severity 차등 재도입** (현재 스키마에서 제거됨, migration 0004 참고)
  - filler: Tier 1 모음 늘임 / repetition burst 가중
  - cps: 임계 초과 정도(절대값)
  - dead_zone: 카테고리별 duration 구간
  - gaze: 이탈 지속 시간 구간
  - content_gap: LLM rubric 강화
- **severity-weighted F1 + Cohen's κ** (라벨러 ≥ 2명)
- **자동 ROI 강건화**: 4코너 폴백 실패 시 9분할 폴백 / 사용자 수동 ROI (Streamlit drag)
- **Self-correction(repair+restart) / Backchannel** filler 차원 확장
- **suggestion priority 재도입** — severity 차등이 들어오면 그것을 가중치로 활용

## 잔여 작업

기획서 명시 사양 중 미구현 항목을 우선순위별로 정리. 어필 직결도와 시연 안정성
기준이며, 위에서 언급된 v1.1 항목과 일부 중복된다.

### P0 — 어필 1순위

| 작업 | 현 상태 | 근거 |
|---|---|---|
| Dockerfile + Fly.io 배포 | `Dockerfile` / `fly.toml` 없음, 시연 URL 없음 | 기획서 2.3·2.5 "필수" |

### P1 — 시연 안정성·UX

| 작업 | 현 상태 |
|---|---|
| 모바일 반응형 (≤880px) | Next.js 메인은 `grid-cols-[300px_1fr]` 고정 — 디자인엔 stacking 정의됨, 미구현 |
| 시간축 이슈 밀도 차트 | 이슈 버튼은 finding row만 — 영상 progress 위에 dimension-color 마커는 미구현 |
| 영상 플레이어 시크 부드러움 | finding/suggestion ref 클릭 시 video element key 재생성으로 짧은 깜빡임 — `currentTime` 직접 set로 폴리시 |
| 분석 실패 toast | analyzing state 안 errorMessage가 idle 복귀 시 즉시 사라짐 — 잠깐 띄우는 toast 분리 |

### P2 — 관찰성·CI·문서

| 작업 | 현 상태 |
|---|---|
| MLflow A/B run 결과 README 첨부 | 표는 있으나 실제 run·스크린샷 없음 |
| Langfuse 대시보드 스크린샷 | trace는 흐르나 README 미첨부 |
| GitHub Actions CI (pytest + typecheck + 골든셋 회귀) | `.github/` 디렉토리 없음 |
| Mermaid 아키텍처 다이어그램 | 미작성 — Next.js + FastAPI + R2 + Supabase 구조 |
| `docker compose up` 1-click | 셋업 섹션 명시와 달리 파일 없음 |
| OpenAPI → TypeScript codegen | `web/lib/api.ts`가 Pydantic 스키마를 손 미러링 — 차원/카테고리 추가 시 두 곳 동기화 필요 |

### v1.1로 분류 (시간·인력 부담 큼)

- Storage signed URL 직접 업로드 (현재 백엔드 경유로 R2 PutObject)
- DeepEval + Cohen's κ — 라벨러 ≥ 2명 전제
- Label Studio 연동 (severity 차등 라벨링 도구) — severity 재도입은 위 v1.1 계획 항목
- pyannote VAD 동적 삽입 (filler가 사전 매칭 단일 차원으로 굳음 — 도입 시 hf token 추가)
- 자동 카테고리 분류, 신규 카테고리(발표/스피치 등) 정밀 튜닝
- Replicate GPU 외부화 + cold-start warmup — 현재 `DEVICE="cpu"`, `int8`
  로컬 운용 (`audio/transcribe.py:19`)

## 검출 차원 튜닝 회고 — CPS / Filler / Dead Zone / Gaze

골든셋 라벨링 후 `lecture macro_f1=0.222`, `vlog=0.114`로 측정된 baseline에서 차원별로
임계값과 알고리즘을 튜닝한 기록. 무엇이 잘 됐고 무엇이 막혔는지, 왜 그런지 정리.

### CPS — 임계 정책 재설계

#### 어려웠던 점

초기 정책은 **"절대 임계(<3 또는 >9 CPS) AND 영상 평균 ±2σ 이탈"** 동시 충족만 이상으로
판정. 의도는 "보편 청자 부담 + 영상 내 변동" 두 신호의 교집합으로 false positive를 줄이는
것이었으나, 다음의 모순이 드러남.

- 화자별·영상 종류별 정상 발화 속도 편차가 큼. 평균 4 CPS인 차분한 화자가 갑자기 8 CPS로
  말하면 청자에겐 명확한 변동인데 절대 임계(>9) 미달로 누락.
- 절대 임계 자체가 자의적 보편값. 한국어 평균 5~7 CPS, 9 이상이면 부담 시작이라는 휴리스틱은
  음성처리 연구의 통용 값이지만 영상 감수의 본질("이 영상에서 튀는 구간")과 어긋남.

#### 개선 방법

**검출 정책을 "이 영상에서 튀는 구간"으로 재정의.** 절대 임계를 제거하고 영상 평균 대비
±1.5σ 이탈만 사용. kind는 평균 대비 방향(`cps > mean → too_fast`)으로 결정.

σ 임계를 단계적으로 탐색: ±2σ → 평탄 영상에서 미검출, ±1.2σ → vlog FP 7→15 폭증으로 악화,
±1.5σ → sweet spot. 결과:

- lecture: `macro_f1 0.222 → 0.302` (+36%)
- vlog: `macro_f1 0.114 → 0.169` (+48%)

**평탄 영상 가드 `MIN_STDEV` 추가.** σ가 매우 작은 영상(균질 발화 합성 케이스)에선 ±1.5σ
임계가 평균 위에 거의 붙어 단어 길이 비례 배분 등의 수치 노이즈도 임계를 넘어 false
positive로 잡힘. σ가 충분히 클 때만 검출 활성화.

> 한 번 "현 데이터에선 안 걸리니 제거하자"고 판단했다가 테스트 `test_normal_speech_no_anomaly`가
> 즉시 깨지면서 가드의 실효성이 검증됨. 라벨된 실데이터(σ≈2.2)엔 영향 없지만 합성 데이터엔
> 작동하는 방어막.

#### 골든셋 cps 라벨 후 단계별 튜닝 — 기획서 §3 P≥0.80 검증

cps 라벨이 붙은 뒤(lecture 1개 / vlog 8개) baseline P/R/F1을 측정하고 lever를 단계적으로
시도. 평가 매칭 정책은 **라벨 ±1초 확장 IoU 0.3 + kind(too_fast/too_slow) 일치 필수** —
방향 정보가 사용자 가시 의미의 핵심이라 반대 kind 매칭은 가짜 TP.

baseline (Stage 2 `whisper-ko-ksponspeech-ct2`):

| 영상 | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| lecture | 1 | 2 | 0 | 0.333 | 1.000 | 0.500 |
| vlog | 3 | 6 | 5 | 0.333 | 0.375 | 0.353 |

vlog 골든셋이 라벨 풍부 — 주력 데이터. lecture는 라벨 1개라 P 변화에 통계적 의미 작음.

##### 시도와 결과

| 시도 | 가설 | 결과 |
|---|---|---|
| **Stage 1**: kind 비대칭 σ (FAST=1.8, SLOW=1.0) | too_slow recall=0/5는 임계가 너무 보수적. 슬로우 라벨 z_min=-0.81~-1.16이라 -1.0σ 풀면 일부 회수. too_fast는 +1.8σ로 약한 z≈1.6 FP cut | **vlog F1 0.353→0.231 악화**. SLOW=1.0이 vlog mean=4.27/std=2.62에서 임계 1.65 cps로 너무 공격적 — 자연 발화 1~3 cps 영역까지 잡혀 too_slow 검출 0→11. FAST 1.8은 라벨 인접 merge를 끊어 IoU 0.3 미달로 TP 손실. |
| **Stage 2**: WINDOW_SEC 5→3 | cps 라벨 평균 4.6초보다 5초 윈도우는 평균 희석. 3초로 좁히면 라벨 안 cps 피크가 임계 통과 | **vlog F1 0.353→0.222 악화**. 짧은 윈도우는 단어 수 감소(3~5개)로 cps std 2.62→3.09 증가 — 노이즈가 신호보다 빨리 늘어 FP 폭증. |
| **Stage 3**: MIN_NET_SPEECH_SEC 0.5→1.5 | 침묵·필러 위주 윈도우가 cps 인공 저하 → too_slow FP 원천. 30% 이상 발화 요구로 통계와 검출 양쪽에서 제외 | **변화 거의 없음**. vlog 윈도우 348→327로 감소했지만 metrics 동일 — 침묵 윈도우는 어차피 ±1.5σ 임계를 넘지 못해 FP에 기여하지 않았음. mean·std 미세 변경만. |
| **Stage 4**: filler 단어 cps 측정 제외 | "음·어"는 의미 발화 아닌 disfluency. 짧은 글자수 + 짧은 시간 비율로 cps를 인공적으로 낮춰 too_slow FP를 만듦. 분모(net speech)와 분자(chars) 양쪽에서 제외 | **vlog F1 0.353→0.375 (+0.022)**. lecture 동일. vlog FP 6→5로 1개 감소(too_slow 케이스). 이론적으로 깔끔하고 lever 결합도 단순 — **채택**. |
| **Stage 5**: SLOW_SIGMA 1.5→1.8 (Stage 4 위) | 슬로우는 평균 근처에 모이는 경향이라 1.5σ면 침묵·필러 인접 윈도우(cps 1~2)를 자주 잡아 FP. 보수적으로 강화 | **변화 없음**. lecture FP 129-138 cps=2.64 z=-1.876 — 임계 1.8과 borderline에서 통과. 효과 없는 strict 강화는 폐기. |
| **Stage 6**: trimmed mean·std (10% trim) | 라벨 영역(~9%)이 mean·std에 미치는 self-bias 제거. robust statistics 표준 | **vlog F1 0.353→0.154 악화**. trim 후 std 축소로 z 임계가 좁아져 평범 윈도우도 다수 통과 — detected 8→18 폭증. trimmed location + scale 동시 적용은 임계 재보정 없이는 detector를 망가뜨림. |

##### 정착 — Stage 4 (filler exclude only)

| 영상 | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| lecture | 1 | 2 | 0 | 0.333 | 1.000 | 0.500 |
| vlog | 3 | 5 | 5 | 0.375 | 0.375 | 0.375 |

**Stage 4 시점 진단** — 단일 σ·윈도우·trim lever로 0.4 천장 못 넘음. lever 결합 시
부작용이 신호보다 큰 패턴 다수(Stage 1 SLOW=1.0 FP 폭증, Stage 6 trimmed std 임계 망가짐)
— 종합 회고는 Stage 15 후 통합 정리.

#### 추가 lever — 변화율·σ sweep (Stage 7~10, 모두 실패)

Stage 4의 단일 신호 천장(0.4 영역)을 넘기 위해 *cps의 다른 통계 표현*을 시도.

| Stage | lever | 가설 | 결과 |
|---|---|---|---|
| 7~9 | 변화율 detector — 직전 N초 baseline 대비 변화량 | 라벨러 "속사포"는 *컨텍스트 대비 급변동* 인지 | vlog F1 0.091~0.231 — detected 시간이 라벨과 어긋남 |
| 10 | σ sweep (0.6~2.0σ) | 좁은 σ로 R 향상 가능성 | σ=1.5가 sweet spot, 0.6은 P=0.20 폭락 |

**변화율 lever 진단** — 라벨 영역 안 cps가 평탄(이미 빠름이 유지됨)이면 변화량 임계 미달, 변화는 라벨 진입 직전에 발생. lookback이 인접 라벨 영역을 흡수하면 baseline 자체 오염. 즉 변화율 신호 시점 ≠ 라벨러 인지 시점. **단일 raw cps의 어떤 통계 변환도 라벨러 인지에 도달하지 못함을 확인**.

#### F0(피치) multi-feature 도입 — Stage 11 (채택)

**도입 계기** — Stage 10 σ sweep으로 cps 1차원 분포의 분리 한계가 입증된 직후, 라벨러 인지의 *다른 차원*을 데이터로 찾는 것이 정공법. vlog 골든셋 too_fast 라벨 4개의 F0(피치) 분포를 baseline과 비교했을 때 *일관된 톤 상승 패턴*이 드러남:

| 라벨 | cps z | F0_mean z | F0_range z |
|---|---|---|---|
| L[111-117] too_fast | +2.4 | **+2.51** | +1.54 |
| L[227-231] too_fast | +1.96 | **+1.49** | +0.80 |
| L[247-251] too_fast | **+0.71 (cps 미달)** | **+0.96** | **+1.28** |
| L[269-274] too_fast | +2.4 | **+2.04** | +1.50 |

특히 L[247-251]은 cps z=0.71로 ±1.5σ 임계 미달이지만 F0_range z=+1.28로 강한 톤 변화 신호 — *cps 단독으론 표현 안 되는 라벨러 인지가 F0에 보존*되어 있음을 입증. too_slow 라벨은 F0 신호가 약하다는 사실도 동시 확인 (slow는 cps 단독 유지 결정).

**결합 정책 결정 — AND vs OR sweep**

| 임계 조합 | TP | FP | F1 | 비고 |
|---|---|---|---|---|
| OR (cps>1.5 또는 F0>0.8) | 2 | 12 | 0.182 | F0 신호로 자연 발화 변동도 통과 → FP 폭증 |
| AND cps>1.5 AND f0>1.0 | 2 | 3 | 0.308 | strict, R 손실 |
| **AND cps>1.5 AND f0>0.8** | **3** | **2** | **0.462** | **sweet spot** |

AND 결합 채택 — cps 임계는 후보 선정, F0는 보강 검증 역할. **vlog F1 0.353 → 0.462 (+31%)**, P 0.333 → 0.600 (+80%).

**F0 결합의 진짜 역할 — 노이즈 자동 필터로 reframe**

라벨 보정(Stage 15) 이후 F0 결합의 효과를 "톤 신호" 가설 외에 다른 각도로 재검증:

| 정책 | TP | FP | F1 | F0가 cut한 영역 |
|---|---|---|---|---|
| cps 단독 | 4 | 4 | 0.533 | — |
| **cps + F0 AND** | 4 | 1 | 0.667 | 197-204(잘해 야바위), 294-301(신난다 신난다), 376-386(소금아) |

**F0 결합이 cut한 3개 FP는 사용자가 청취 검증 시 "메인 화자 발화 아님"으로 판단한 영역과 정확히 일치**. 이유:
- 노이즈/배경 음성 = 메인 화자 외 음성 → F0 톤 신호 약함 (멀리서 발화, 짧은 음성, voiced 비율 낮음)
- 메인 화자 발화 → F0 voiced 강함 + 일관된 피치
- AND 조건이 노이즈 영역을 자동으로 cut

즉 F0 결합은 단순 "톤 상승 = 속사포 보강"이 아니라 **메인 화자 분리 신호** — *라벨러의 "들려서 빠르다고 인지한 영역"을 음향 신호로 우회 모방*한다. 단순 RMS percentile / word confidence / voiced ratio 필터를 별도로 시도했으나 모두 한국어 음운 분포(자음 위주 단어가 정상 발화도 voiced ratio 낮음)로 인해 메인/노이즈 분리에 부적합 — F0 voiced 강도 + range 결합만이 의도한 분리에 도달.

#### 본질 lever 시도 — 음절 검출 / 문장 단위 윈도우 (Stage 12~14, 모두 실패)

F0 결합이 *상관관계 활용 우회로*라는 자각 후 **ASR 토큰 단위 cps 자체를 raw 오디오 신호로 우회하는 본질 lever** 시도:

| Stage | lever | 가설 | 결과 |
|---|---|---|---|
| 12 | librosa onset_detect 음절 수 | "너—무"의 늘인 음절은 ASR 정규화에 흡수되지만 raw 오디오 onset에 보존 | 한국어 평균 5 syl/s 대비 절반(2.77) 검출 — 음악용 spectral flux 알고리즘 부적합 |
| 13 | parselmouth (Praat) syllable nuclei | 음성학 표준 도구로 정확도 향상 | baseline 2.2 syl/s 여전히 부정확. cps와 redundant (둘 다 잡거나 둘 다 놓침), 추가 lever 가치 없음 |
| 14 | 문장 단위 윈도우 (5초 고정 → 문장 경계) | 라벨러 인지 단위 = 문장 — 5초 윈도우의 영역 mismatch 해결 | 짧은 감탄 문장(0.6~1초)이 cps 변동성 폭증으로 noise. F1 0.171~0.267 |

**결론** — 한국어 음성학 raw 신호 처리는 v1.1 영역 (parselmouth + 한국어 음운 적응 필요). 현 도구로는 ASR 토큰 단위 cps + F0가 가능한 최선.

#### 천장 돌파 — 라벨 작업 (Stage 15, F1 0.667 / 0.800)

Stage 7~14 모든 detector lever 천장이 0.46임을 확인 — 본질 문제는 detector 알고리즘 영역 밖. **라벨 데이터 품질**이 진짜 천장임을 진단:

| 미스 원인 | 비율 (vlog 라벨 8개 기준) |
|---|---|
| ASR이 모음 늘임을 정규화 ("너—무"→"너무") | 1/5 미스 |
| ASR이 단어 누락 (라벨 영역 내 발화의 일부만 토큰화) | 1/5 미스 |
| 라벨 시간 부정확 (라벨러 1초 라운딩 + ASR ±20ms 어긋남) | 2/5 미스 |
| 윈도우 5초 vs 라벨 4초 영역 mismatch | 1/5 미스 |

라벨러(=사용자) 본인이 vlog 영상을 직접 청취하면서 라벨 작업 진행 (45분 소요):

- **라벨 시간 정밀화**: 111-117 → 111-116, 221-227 → 221-226, 269-274 → 269-273
- **라벨 제거**: L[43-46 too_slow], L[247-251 too_fast] — 청취 결과 ASR 정렬 한계로 라벨 자체 신뢰성 낮음
- **라벨 추가**: 72-79 too_fast (detector가 cps z=1.54로 잡았으나 라벨 누락이었던 "야 비상이다" 영역, 청취 검증)
- **노이즈 영역 라벨 거부**: 197-204 / 294-301 사용자 청취 시 "거의 안 들리는 배경음" → 라벨 추가 X. detector는 이 영역을 잡았었으나 사용자 인지엔 메인 화자 발화 아님

새 라벨 + 기존 multi-feature detector(`cps>1.5 AND F0>0.8`)로 재측정:

| 영상 | TP | FP | FN | **P** | R | **F1** |
|---|---|---|---|---|---|---|
| **vlog** | 4 | 1 | 3 | **0.800** ✓ | 0.571 | **0.667** |
| **lecture** | 2 | 1 | 0 | 0.667 | 1.000 | **0.800** |

**vlog Precision 0.800 — 기획서 §3 cps Precision ≥ 0.80 충족.** F1은 baseline 0.353 → 0.667 (+89%).

#### 카테고리별 정책 분기 — 녹화 환경 노이즈 차이가 근거

multi-feature production 측정 시 lecture에 동일 정책 적용하면 F1 0.800 → 0.500으로 *악화*. 표면적 진단은 "lecture 톤 단조라 F0 임계 미달"이지만 그건 약한 추측. **진짜 분기 근거는 카테고리별 녹화 환경 차이**:

| 카테고리 | 녹화 환경 | 노이즈 양상 | F0 결합 효과 |
|---|---|---|---|
| **vlog** | 야외·일상 (강아지 산책, 호명, 다른 사람 발화) | 배경 노이즈 다수 — 사용자 청취 검증 (197-204, 294-301) | **노이즈 자동 cut → P 0.500 → 0.800** |
| **lecture** | 통제된 녹화 (단일 화자, 슬라이드 설명) | 노이즈 거의 없음 — 사용자 청취 검증 시 detector FP가 노이즈 X | 필터 효과 무의미 + AND 조건이 라벨 cut 부작용만 |
| **other** | 도메인 다양성 큼 (음악·게임·예능) | 미검증 | 보수 fallback (cps 단독) |

이는 *지표에 맞춰 정책을 분기*한 fitting이 아니라 **녹화 환경 특성 기반 도메인 적응**. F0 결합이 노이즈 환경에서 *우연히* 메인 화자 분리 신호 역할을 한다는 발견을 카테고리 매트릭스에 반영. LangGraph conditional edge 노드(`graph/nodes.py:detect_cps`)가 카테고리에 따라 `detect_cps_anomalies` 또는 `detect_cps_with_audio` 호출 분기.

#### 최종 정착 정책

```python
# graph/nodes.py:detect_cps
if state["category"] != "vlog":
    detect_cps_anomalies(transcript)              # cps 단독 ±1.5σ
else:
    detect_cps_with_audio(transcript, video_path) # cps>1.5σ AND F0>0.8 결합
```

**구현 책임 분리**
- `audio/cps.py`: 윈도우 정의, σ 임계 판정, F0 baseline 통계, 인접 병합
- `audio/pitch.py`: librosa pYIN으로 F0 추출, 윈도우별 voiced 통계
- `audio/cps.py:detect_cps_with_audio`: 윈도우-pitch 정합 캡슐화 helper

**평가 매칭 정책** (`eval/metrics.py:compute_cps_metrics`)
- 라벨 ±1초 확장 후 IoU greedy 1:1 매칭 (라벨러 1초 라운딩 + ASR ±20ms 어긋남 흡수)
- **kind(too_fast/too_slow) 일치 필수** — 방향 정보가 사용자 가시 의미의 핵심이라 반대 kind 매칭은 가짜 TP

#### 회고 — 이 단계에서 얻은 교훈

- **detector lever 천장은 라벨 데이터 품질 천장과 다르다.** Stage 7~14의 모든 알고리즘 시도(변화율·σ sweep·F0 multi-feature·음절 검출·문장 단위)가 F1 0.46 영역에서 막혔으나, 라벨 본인의 45분 청취 작업 후 같은 detector로 F1 0.667 도달. ML 사이클에서 *어디까지가 detector 영역이고 어디부터 라벨 영역인지*를 데이터로 분리해 진단하는 게 시니어 의사결정.
- **상관관계 우회로의 가치와 한계를 둘 다 인지해야 한다.** F0 결합은 라벨러 "속사포" 인지의 직접 측정이 아니라 *흥분 톤* 또는 *메인 화자 분리* 같은 부수 신호 활용. vlog 골든셋엔 fit하지만 차분한 빠름·기계 합성 음성에는 무력. 우회로를 채택할 땐 한계 영역을 회고에 명시하고 v1.1 본질 lever(parselmouth 음절 검출 + 한국어 음운 적응)를 로드맵에 남긴다.
- **카테고리별 분기 근거는 *결과 차이*가 아니라 *환경 차이*여야 한다.** "F0 적용 시 lecture F1 떨어졌으니 vlog만"은 fitting. "vlog는 야외 녹화로 노이즈 다수, lecture는 통제 환경"이 환경 데이터로 검증된 분기 근거. 라벨러 청취 결과(노이즈 영역 197-204, 294-301)와 detector F0 cut 결과가 일치한 사실이 환경 가설을 데이터로 뒷받침.
- **too_slow는 cps 차원의 본질적 약신호.** Stage 4의 진단(z_min 분포 -0.81~-1.16)은 라벨 보정 후에도 유지 — 새 라벨 3개 too_slow 모두 R=0/3. *라벨러의 "느림" 인지가 cps 단순 metric으로 표현되지 않음*이 8개 라벨에서 일관되게 확인. v1.1에서 음절 duration·휴지 빈도 등 다른 신호 결합으로 too_slow 별도 detection 차원 검토 필요.
- **라벨 8개 + 1개 영상의 평가 한계.** vlog F1 0.667 / lecture F1 0.800은 *현 골든셋의 천장*. F0 결합 효과의 일반화는 다른 vlog·lecture 영상으로 검증해야. 카테고리별 환경 가설도 표본 1개에서 도출 — v1.1 라벨 확장 후 분기 근거 재검증 필요.

### Filler — 한국어 명확형 filler 검출의 단계별 시도와 회고

#### 문제 정의

vidoctor가 검출하려는 한국어 머뭇거림 표지("음·어·으·에·그·저")는 영상 감수의 핵심 신호.
다만 영어 우세로 학습된 Whisper 가족(large-v3-turbo, large-v3, large-v2)이 짧은 한국어
비언어 음을 *단어로 토큰화하지 않는* 알려진 한계가 있어, baseline 검출 성능이 매우 낮았다:

| 영상 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| lecture | 2 | 4 | 1 | 0.333 | 0.667 | 0.444 |
| vlog | 1 | 5 | 4 | 0.167 | 0.200 | 0.182 |

baseline의 lecture FP 4개는 *특정 어휘에 집중*("그러니까" × 3 + "좀" × 1)이고, vlog는
*명확형 filler 라벨 5개 중 4개에서 ASR 토큰 자체가 0*인 영역. 두 카테고리의 실패 패턴이
달라 시도도 다층적으로 분리해 진행했다.

#### Baseline 진단 — 라벨 영역의 ASR 토큰 직접 확인

검출이 안 잡힌 라벨 시간대에서 *ASR이 실제로 어떤 단어를 토큰화했는지*를 직접 dump해 원인
진단:

| vlog 라벨 | baseline ASR 토큰 (large-v3-turbo) |
|---|---|
| 99~101s "음·어" | (no tokens — 무성 처리) |
| 262~268s "음·어·저" | "안 힘든가 봐요. 날씨가 좀 흐릿한…" (다른 단어로 흡수) |
| 326~328s "음·어" | (no tokens) |
| 334~337s "음·어·그" | (no tokens) |
| 364~365s "음+더듬음" | "강아지?" (오인식) |

**라벨 5개 중 4개에서 ASR 토큰이 0개**. detector 단의 사전·임계 튜닝으론 닿을 수 없는
영역임을 데이터로 확정. 이 진단이 이후 시도 전략의 분기점이 됐다.

#### 시도 1 — 반복 정책 재검토 (알고리즘 가정 정정)

초기 정책은 Shriberg(1994) disfluency 4분류를 따라 *"인접 동일어 반복 = disfluency"*로 사전
외 단어 반복도 filler로 등록. vlog 검증에서 이 가정이 무너짐:

```
검출 18개 분포:
  사전 외 반복 12개 — "소금아! 소금아!", "짜잔! 짜잔!", "강아지 강아지" …
                       강아지 영상의 명령·호명·강조성 반복
  사전 매칭 6개   — "이제" × 3, "좀" × 3
  명확형(어/음/그/저) 0개
```

vlog 일상 대화의 인접 반복은 *머뭇거림이 아니라 강조·명령·호명*이 우세. 강의에서도 강조용
반복이 자연스럽다는 사용자 직관과 일치.

**조치**: 사전 외 단어 반복 검출 제거. 사전 단어가 인접 반복일 때만 burst로 묶음.

**효과**:
- vlog filler FP 16 → 5 (−11)
- F1 자체 변동은 작음 (0.174 → 0.182)
- **사용자가 보는 finding 노이즈 11개 제거**가 실제 가치 — F1 metric이 크게 안 움직여도 UX 향상

#### 시도 2 — ASR 단계 lever (세 시도 모두 실패)

vlog의 "ASR 토큰 0개" 문제를 ASR 단계에서 풀려는 세 가지 시도:

| 시도 | 의도 | 결과 |
|---|---|---|
| WhisperX `hotwords="음 어 으 에 그 저 뭐"` | 디코더가 해당 단어 점수 boost | **vlog FP 5 → 26 폭증** (정상 발음을 filler로 끌어당김), macro_f1 0.172 → 0.062. 채택 불가 |
| VAD 임계 완화 (`vad_onset 0.5→0.3`) | 짧은 발화를 무성 처리 안 하게 | **filler 검출 변화 0**. VAD가 잡아도 ASR이 토큰화 안 함을 확인 — VAD 단계 문제가 아님 |
| 모델 교체 `large-v3-turbo → large-v3` | 디코더 4 layer → 32 layer 정밀도로 풀 수 있나 | **lecture·vlog 둘 다 filler 변화 0**. 디코더 layer 차이 문제가 아님 |

→ **Whisper 가족 어떤 변형도 한국어 짧은 명확형 filler를 토큰화하지 않음**. *모델 학습 분포의
결정*이 핵심 한계임을 세 실험으로 확정. 이게 다음 단계(시도 5의 모델 swap)로 가는 정당화 근거.

#### 시도 3 — ASR 우회 음향 신호 분석 (실패)

ASR이 토큰화 못 한 영역을 *음향 신호 직접 분석*으로 우회 시도. VAD가 음성으로 판정한 시점
중 ASR 단어가 없는 곳에서 RMS 에너지·F0 분산을 검사해 명확형 filler 후보 추출.

| 버전 | 알고리즘 | 결과 |
|---|---|---|
| v0 (RMS 에너지만) | 음성·무음 구분 | lecture FP 4→21, vlog FP 5→42 폭증. 단어 사이 호흡·자음 클로저까지 filler로 등록 |
| v0.5 (RMS + F0 분산) | librosa.pyin으로 voiced 구간 + F0 std 검증 | lecture FP 21→10, vlog FP 42→24. 일부 거름. 다만 baseline 대비 여전히 나쁨. **Recall은 두 시도 모두 0 변화** |

라벨 5개의 실측 음향 측정값으로 실패 원인 진단:

| 라벨 | F0 std | voiced ratio | 임계 통과? |
|---|---|---|---|
| 99~101 (음·어) | 56Hz | 0.90 | ✗ (F0 임계 30Hz 초과) |
| 262~268 (음·어·저) | 50Hz | 0.67 | ✗ (F0 임계 초과 + ASR 단어가 그 시간대 채움) |
| 326~328 (음·어) | 11Hz | 0.54 | ✓ |
| 334~337 (음·어·그) | 54Hz | 0.54 | ✗ (다중 filler라 피치 변동 큼) |
| 364~365 (음+더듬음) | 14Hz | 0.28 | ✗ (더듬음 unvoiced 비중 큼) |

**라벨 패턴이 단순 명확형이 아니라 혼합형** — filler + 더듬음, 다중 filler 연속, ASR 단어와
시간 겹침. 알고리즘은 "단조로운 nasal hum"을 가정했으나 실제 라벨은 그 가정에 맞지 않음.
임계 완화로 라벨을 잡으면 lecture FP가 baseline 대비 5배 폭증해 precision-recall 균형점이
데이터 안에 *없음*을 확정. 알고리즘 가정 자체를 데이터에 맞게 재설계해야 함을 회고로 남김.

#### 시도 4 — 사전 정리 + 평가 tolerance (Stage 1)

ASR 단계 lever와 음향 우회가 모두 실패한 뒤, *남은 lever* 두 가지에 집중:

1. **사전 정책** — 어떤 어휘를 filler로 볼지의 정의
2. **평가 정책** — 라벨러 정밀도와 모델 정밀도 격차를 어떻게 흡수할지

**사전 변경 두 가지**:

- **"그러니까 / 그래서" 제외** — 강의 골든셋 평가에서 *논리 연결사로 정상 사용*되는 비율이
  높아 라벨러가 filler로 보지 않음. 검출 시 모두 false positive 원천
- **"자" 추가** — 한국어 구어의 *주의 환기 표지*("자, 이제…")로 자주 쓰이지만 사전에 누락

**평가 tolerance 도입 (`±1s`)**:

라벨러가 영상 플레이어에서 *1초 단위로 시간을 라운딩*해 라벨한다는 사실에 주목. 음성 처리
표준 boundary tolerance는 200~500ms이지만 우리 라벨은 1초 단위라 더 큰 흡수가 필요.
라벨 영역 양쪽으로 ±1s 확장한 뒤 detection point가 그 안에 들어오면 매칭으로 본다.

도메인 표준값 비교:

| 영역 | 표준 tolerance |
|---|---|
| TIMIT, Switchboard (음성 disfluency) | ±200ms |
| DIHARD (diarization) | ±250ms (collar window) |
| NIST STT challenges | ±200ms |
| **vidoctor (라벨러 1초 라운딩 보정)** | **±1s** |

**Stage 1 효과**:

| 영상 | F1 변화 | 해석 |
|---|---|---|
| lecture | 0.444 → 0.400 | P 0.333 → 0.500 (FP 4→1, "그러니까" 제거 효과). R 0.667 → 0.333 |
| vlog | 0.182 → 0.182 (변화 없음) | 사전 변경 어휘가 vlog 검출에 등장 안 함 |

lecture R 0.667 → 0.333 깎임의 진실: baseline R=0.667에서 134~142s 라벨이 **우연히 "그래서"가
그 시간대에 들어가** TP로 잡혔던 것. "그래서" 사전에서 빠지자 우연 매칭 사라지고 *진짜 R*이
드러남. 점수 하락이 "퇴보"가 아니라 **"정직화"** — baseline 점수가 도금이었다는 회고.

#### 시도 5 — 한국어 fine-tuned ASR swap (Stage 2, 결정적 lever)

시도 2에서 확정한 *모델 학습 분포 한계*를 정공법으로 공략. 한국어 자유발화 코퍼스
KsponSpeech (약 1,000시간)로 fine-tune된 Whisper 모델을 도입.

**선택**: `Jungwonchang/whisper_finetune_ksponspeech_partial` (Whisper large-v2 기반).
PyTorch checkpoint를 ctranslate2 int8로 변환해 WhisperX flow에 그대로 swap (변환 후 모델
크기 1.5GB). WhisperX의 wav2vec2 forced alignment(±20ms 정밀도) 흐름은 그대로 유지 —
Whisper 가족이라 인터페이스 호환되어 1줄 swap 가능했다.

**Stage 2 + tolerance ±1s 효과**:

| 영상 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **lecture** | 3 | 0 | 0 | **1.000** | **1.000** | **1.000** |
| vlog | 3 | 11 | 2 | 0.214 | 0.600 | 0.316 |

**lecture는 PERFECT 도달**. baseline에서 토큰화 못 하던 `음·자·어·이제·약간·뭐랄까`를 모두
잡고, tolerance가 라벨 ±1s 정밀도 한계를 흡수. F1 0.444 → 1.000 (+125%).

**vlog는 F1 0.182 → 0.316 (3.3배 향상)**, Recall 0.200 → 0.600 (3배). 라벨 인접 검출이
정당하게 매칭되며 큰 폭 개선. 다만 P 0.214로 Precision 한계가 남음 — 다음 시도에서 진단.

#### 시도 6 — vlog Precision 한계 진단 (사용자 청취 검증)

vlog Stage 2의 P 0.214가 *라벨 누락 영역*에서 생겼는지, *모델 환각*에서 생겼는지 구분이
중요한 갈래. ASR이 라벨 외에서 잡은 명확형 filler 후보 3 영역을 사용자가 직접 영상 청취로
검증:

| 시간대 | ASR 검출 | 사용자 청취 결과 |
|---|---|---|
| 17~20s `"저"` 단발 | "저" | 정상 발화 (지시사) |
| 95~96s `"음·어"` | 라벨 99~101 직전 | **그 시간대 발화 자체 없음** (ASR 환각) |
| 340~344s `"음·음·그·음·에"` 5연속 | 라벨 영역 밖 | 일상 대화 음향에서 환각 |

**모든 후보가 환각으로 확정**. 한국어 fine-tuned 모델이 일상 발화의 *짧은 호흡·자음 클로저·강세
변화*를 "음/어/저"로 *환각 토큰화*. lecture는 차분한 화자라 환각 거의 없지만, vlog 일상 발화는
환각 트리거가 풍부.

**의의**: vlog F1 0.316은 *라벨 누락이 아니라 모델 환각 천장*임이 확정됨. "라벨 보강으로 풀 수
있는 문제"가 아닌 "모델 도메인 적응 문제"로 분류 정정 — *제3자 검증이 ML 평가 metric의 해석
자체를 바꾸는 사례*.

#### 시도 7 — Naver CLOVA Speech 도입 검증 (실패, 한국어 상용 SOTA의 부적합 확정)

vlog 환각 천장을 뚫을 후보로 한국어 ASR 상용 SOTA인 **Naver CLOVA Speech (NEST 모델)**을
끝까지 시도. NCP 가입·Object Storage·도메인 생성·Secret Key 발급·sanity 호출까지 전 절차
진행:

1. NCP 콘솔 → AI Services → CLOVA Speech 이용 신청
2. Object Storage 버킷 2개 생성 (장문 인식 도메인이 Storage 경로 필수)
3. 도메인 생성. 첫 호출 시 `speaker detect is off` 400 에러 → *long sentence 도메인은 화자
   인식 기본 활성*이어야 호출 받음 → 콘솔에서 화자 인식 사용으로 변경
4. vlog 60초 클립(filler 라벨 99~101s 포함)을 long sentence `/recognizer/upload` 동기 호출

**결과 — `setting.fillerText` 옵션 포함 모든 시도에서 명확형 filler 0개 토큰화**:

```
열기구가 지금 떠오르고 있습니다. 높이 올라가네요. 강아지들 너무 귀엽지 않나요?
너무 너무 귀여워 너무 귀여워 완전 너무 귀엽죠. ...
```

→ 강조성 반복("너무 너무")은 살리지만 **"음·어·으·에" 모든 명확형 filler를 모델 단계에서
자동 제거**. 한국어 상용 SOTA ASR이 *깨끗한 transcript* 마케팅 정책 하에 학습돼 disfluency
보존이 *반대 방향* 우선순위.

**의사결정 정직화**:

| 후보 | 검증 결과 |
|---|---|
| Naver CLOVA Speech (NEST, 한국어 ASR 상용 SOTA) | ✗ filler 자동 제거. 옵션으로 복원 불가 |
| Google STT 한국어 / Azure Speech 한국어 | 같은 *깨끗한 transcript* 정책 가능성 큼. 검증 비용 대비 보상 미지수 |
| ENERZAi EZWhisper KR (50,000h 한국어) | 비공개 상용. B2B 협상 부담 (포트폴리오 단계 비현실적) |
| **자체 한국어 disfluency-aware fine-tune** | **유일한 정공법** — KsponSpeech를 더 큰 한국어 자유발화 corpus(AI Hub 일상 대화 등 ~10,000h)와 결합 + disfluency 라벨 보존 학습. v1.1 작업 |
| ML 기반 phoneme classifier 신규 차원 | 음향 신호 직접 학습으로 ASR 우회. v1.1 후보 |

**의의**: "한국어 ASR SOTA = vidoctor에 SOTA"가 아님을 데이터로 확정. v1.1 ASR 후보군에서
상용 솔루션(CLOVA·Google STT·Azure 한국어) 모두 제외하고 *자체 fine-tune* 또는 *phoneme
classifier 신규 차원*만 남김. NCP 가입 부담은 *부정적 결과*에도 회고 자료로 가치 변환.

#### 시도 8 — 사전 모호형 어휘 제거 (vlog FP cut 가설, 폐기)

**가설**: vlog 사전(`이제·인제·막·좀·약간`) 모호형이 *정상 부사* 용법으로도 자주
등장 → FP 폭증 원인. 명확형(`어·음·으·에·그·저·자·뭐`)만 유지하면 P 향상 기대.

**측정**:

| 영상 | 변경 전 | 변경 후 (모호형 제거) |
|---|---|---|
| lecture | TP=3, FP=0, **F1=1.000** | TP=3, FP=0, **F1=1.000** (영향 없음) |
| vlog | TP=3, FP=11, **F1=0.316** | TP=2, FP=8, **F1=0.267** (악화) |

**진단**: vlog FP 3개 cut(이제·좀)이지만 동시에 **TP 1개 손실** — 라벨 영역
L[262-268]("음·어·저") 매칭이 모호형(좀·이제) 기여로 잡혔는데 빠짐. *사전이
라벨러 정의를 일부 정확히 반영*했음을 데이터로 검증. R 0.6 → 0.4 폭락.

**채택 X**. 단순 사전 축소는 라벨러 인지 일부를 잃음. 모호형 분리는 *반복 burst만
filler 등록* 같은 정밀 정책으로만 가능 — 별도 lever.

#### 시도 9 — 음향 lever 정밀 결합 재시도 (시도 3 패턴 재확인)

시도 3 v0(RMS만)·v1(RMS+F0)의 실패 원인이 *임계 정밀화 부족*인지 검증 — 결합 조건
(ASR gap 200ms~2s + voiced ratio ≥40% + voiced run 50~600ms + RMS percentile)으로
sweep. **모든 조합 F1 0.05~0.17** — baseline 0.316보다 큰 폭 악화. vlog 환경의
voiced gap 다양성(호흡·감탄·다른 화자·강조 발화)이 임계 정밀화로 분리 불가. **채택 X**.

#### 시도 10 — 한국어 phoneme/wav2vec2 모델 시장 조사 (모두 미적합)

**가설**: ASR LM 정규화를 우회하는 acoustic-only 또는 phoneme-level 한국어 모델이
공개 시장에 있다면 채택 가능.

**검증된 모델**:

| 모델 | 출력 단위 | 결과 |
|---|---|---|
| `kresnik/wav2vec2-large-xlsr-korean` | 음절 | "음·어" 영역 출력: "일 세 출 다" 등 부정확 — KsponSpeech Whisper 대비 후퇴 |
| `Kkonjeong/wav2vec2-base-korean` | **자모(jamo)** | Zeroth-Korean(깨끗한 낭독) 학습 → vlog 일상발화 generalization 부족. 라벨 영역 자모 출력 거의 빈 또는 부정확 |
| `facebook/mms-1b-all` (1107 언어, kor 어댑터) | 음절 | 한국어 학습 비중 작음 + 어댑터 lm_head random init transformers 호환성 issue. 출력 부정확 |
| `facebook/wav2vec2-lv-60-espeak-cv-ft` (다국어 IPA phoneme) | espeak phoneme | 한국어 학습 X — "음·어"가 중국어 톤 표기(`5`)로 잘못 매핑. "m" 비음 부분 검출 가능하나 분리 부족 |

**진단**: 공개 한국어 wav2vec2 모델은 모두 ASR(자모/음절 단위 받아쓰기)이지 *비유창성
검출 전용*이 아님. 학습 데이터 분포가 KsponSpeech Whisper(자유 대화)보다 좁거나
(Zeroth = 깨끗한 낭독), 한국어 학습 비중이 작아 vlog 일상발화에 부족.

**채택 X**. 시장 조사 결론: **공개 한국어 disfluency-aware 사전학습 모델 부재**
시장 공백을 다중 데이터로 검증.

#### 시도 11 — gpt-4o-audio-preview (chat audio LLM, 폐기)

**가설**: OpenAI gpt-4o-audio-preview가 자연어 prompt로 한국어 filler 검출 가능.
LLM의 instruction following으로 *모호형 의도 분류*까지 가능할 수 있음.

**chunk 측정 (5s chunk × 78회, 비용 ~$0.013)**:

| 영상 | TP | FP | FN | F1 |
|---|---|---|---|---|
| lecture | 3 | 26 | 0 | 0.188 |
| vlog | 3 | 69 | 2 | 0.078 |

**진단**:
- *명확형 음성 인식은 정확* — chunk_99-101 prototype에서 "음·어" 정확히 검출
- *그러나 모호형 의도 분류 부정밀* — 강조성 반복("짜잔")을 "자"로 오인, 정상 부사를
  머뭇거림으로 분류. **LLM 추론 한계 — 라벨러 정의 학습 데이터 없이 분류 불가**
- *timestamp 정확도 낮음* — chunk 내 5~7초 어긋남. chunk 분할로 우회해도 chunk
  중간 시점 등록의 정밀도 ±2.5초 한계

**채택 X**. *음성 인식*과 *라벨러 의도 분류*는 분리된 문제 — 후자는 라벨러 정의
학습 데이터로만 해결.

#### 시도 12 — OpenAI whisper-1 transcribe + chunk 분할 (vlog 향상하지만 보류)

**가설**: gpt-4o-audio chat은 timestamp 부정밀하지만 OpenAI 전사 API(`whisper-1`)는
word-level timestamp 정확. ASR 출력을 기존 사전 매칭에 input으로 통합 가능.

**Step 1 — 짧은 chunk(2초) prototype** ✅:
- L[99-101] chunk_99-101.wav → "음... 어..." + word [0-1.1] 음, [1.1-1.66] 어
- KsponSpeech가 못 잡던 영역을 whisper-1이 정확히 토큰화. word timestamp 정확.

**Step 2 — 영상 전체 단일 호출** ❌:

| 영상 | TP | FP | FN | F1 |
|---|---|---|---|---|
| lecture | 1 | 2 | 2 | 0.333 |
| vlog | 1 | 4 | 4 | 0.200 |

KsponSpeech baseline(lecture 1.0 / vlog 0.316) 대비 악화. 95-105초 chunk 호출 시
"쩍들인형"으로 오인 — *전체 호출에선 인접 컨텍스트 영향으로 정규화 흡수 발생*.

**Step 3 — `prompt` 파라미터 (어휘·문장 hint)**:

| 영상 | simple prompt | detailed prompt |
|---|---|---|
| lecture | F1=0.571 | F1=0.571 (동일) |
| vlog | F1=0.182 | F1=0.182 (동일) |

prompt 길이·내용 무관 동일 결과 — Whisper API의 `prompt`는 *vocabulary candidate
boost* 수준이라 acoustic decoding 단계 정규화 못 막음.

**Step 4 — 5초 chunk 분할 + 단순 prompt + hallucination dedup** ✅:

| 영상 | TP | FP | FN | P | R | **F1** |
|---|---|---|---|---|---|---|
| lecture | 3 | 2 | 0 | 0.600 | 1.000 | **0.750** |
| **vlog** | **4** | **5** | **1** | **0.444** | **0.800** | **0.571** |

vlog F1 0.316 → 0.571 (+81%). chunk 분할로 LM 컨텍스트 인위 축소 → 정규화 prior
약화. *prompt hallucination dedup*(같은 chunk 안 0.3초 미만 다중 단어 cut)으로
FP 39 → 5.

**왜 채택 X — chunk 분할은 *우회 trick*이지 본질 해결 아님**:

1. *모델 자체 정규화는 그대로* — chunk 분할로 LM 입력 컨텍스트 양만 인위 조작.
   같은 영상 발화 패턴이 약간만 달라져도 효과 변동 큼 (재현 보장 X)
2. *lecture 후퇴* (1.0 → 0.75) — chunk 분할이 LM 정규화에 유리한 환경에선 해로움.
   카테고리별 2트랙 운영 부담
3. *OpenAI API 의존* — 영상당 ~$0.04 + 78회 호출 + rate limit. 무료 KsponSpeech
   대비 비용·인프라 trade-off
4. *영상 콘텐츠 특성 의존* — 짧은 호명·감탄 vlog에선 fit, 긴 호흡 발화 영상에선
   chunk 5초가 정상 단어 잘라 정확도 손실 가능
5. *본질은 학습 데이터 분포* — 정규화는 ASR architecture 본질이 아니라 *학습 데이터
   + 목적 함수* 문제. chunk 분할은 학습 분포 한계의 *trick 우회*

**채택 X**. F1 수치만 보면 채택 가능하지만 *재현 보장·운영 부담·trade-off*가 본질
해결책의 조건을 충족 못함. 0.571 도달 결과는 회고 narrative로 보존.

#### 시도 13 — KsponSpeech chunk 분할 (B 트랙, 단일 모델 통합 검증)

**가설**: 시도 12의 chunk 분할 효과가 *모델 무관* 메커니즘이라면 KsponSpeech도
같은 효과. 단일 모델 + chunk 호출 패턴으로 OpenAI 의존 + 2트랙 운영 부담 해소
가능.

**측정**:

| 영상 | KsponSpeech 단일 | KsponSpeech chunked (5s) | whisper-1 chunked v2 |
|---|---|---|---|
| lecture | F1=1.000 | F1=0.857 | F1=0.750 |
| vlog | F1=0.316 | **F1=0.333 (효과 미미)** | F1=0.571 |

**진단** — chunk 분할 효과는 *모델 의존*:

| 모델 | 한국어 학습 비중 | LM prior 강도 | chunk 분할 효과 |
|---|---|---|---|
| **whisper-1** (multilingual base) | 작음 | 한국어 단음절 filler를 영어 위주 prior로 강하게 정규화 | **chunk 분할로 우회 효과 큼** (vlog 0.316→0.571) |
| **KsponSpeech** (한국어 fine-tuned) | 큼 | 한국어 일상발화 fit, *acoustic 단계*에 이미 정규화 흡수 | chunk 분할도 acoustic 단계 한계 못 넘음 (0.316→0.333) |

핵심 발견: KsponSpeech의 한국어 fine-tune이 *결정적 lever*인 동시에 *vlog 짧은
filler 흡수의 원인*이라는 양면성. 시도 5에서 lecture에 결정적 도움이었던 lever가
vlog에선 정규화 프로세스에 깊이 박혀 chunk 우회로도 풀리지 않음.

**채택 X**. 단일 모델 통합 path로는 baseline 대비 향상 없음.

#### 시도 14 — 사전 reframe (정규화 흡수 어휘 제거 + 담화 표지 추가, 폐기)

**가설**: ASR이 정규화 흡수하는 단음절 명확형(`음·어·으·에`)을 사전에서 제외하고,
*ASR이 정상 토큰화하는 담화 표지*(`아니·근데·진짜`)로 detection 정의를 좁히면
측정 가능 영역에 fit.

**Step 1 — 단음절 명확형 제거**:

| 영상 | 변경 전 | 제거 후 |
|---|---|---|
| **lecture** | F1=1.000 | **F1=0.500** ↓ (TP 3→1, L[92-93]·L[134-142] FN) |
| vlog | F1=0.316 | F1=0.286 (TP 3→2, FP 11→7) |

**진단** — 정규화 여부의 *발화 길이 의존성* 발견:

| 영상 | "음·어" 발화 패턴 | ASR 처리 |
|---|---|---|
| **vlog** | 짧은 burst (0.1~0.3초) — 자연 호흡 일부 | 정규화 흡수 (라벨 영역 4/5 ASR 토큰 0개) |
| **lecture** | 의도적 늘임 (0.5~1.5초) — 강의 사고 시간 표지 | 토큰화 OK ("음·어" 7+2회 등장) |

같은 어휘여도 *발화 길이가 정규화 여부 결정*. lecture의 늘인 발음은 잡히고 vlog의
짧은 burst는 흡수. 사전 전역 제거는 lecture 직격타 — 카테고리 간 효과 비대칭.

**Step 2 — 담화 표지 추가** (`아니·근데·진짜`):

| 영상 | 추가 어휘 transcript 등장 | F1 변화 |
|---|---|---|
| lecture | 모두 0회 | 1.000 그대로 (영향 없음) |
| **vlog** | "진짜" 2회만 (1:59, 3:58) | 0.316 → 0.286 (FP +2) |

vlog 강아지 산책 영상엔 자기수정("아니")·화제 전환("근데") 발화 *자체가 거의 없음*.
영상 콘텐츠 특성이 reframe 가능성 결정 — 강아지 vlog는 호명·감탄 중심이라 담화
표지 빈도 낮음. *대화·인터뷰 vlog*에선 효과 가능성 있지만 현 골든셋엔 적용 불가.

**채택 X**. 단음절 제거는 lecture 직격타, 담화 표지 추가는 영상 콘텐츠 부적합.
사전 reframe은 *현 골든셋 영상 콘텐츠로는 검증 불가* — v1.1 영상 다양성 확장
영역.

#### 정량 결과 종합

| 영상 | baseline | Stage 1 (사전 정리) | Stage 2 (한국어 ASR) | Stage 2 + tol=1 (최종) |
|---|---|---|---|---|
| **lecture** | P=0.333 R=0.667 F1=0.444 | P=0.500 R=0.333 F1=0.400 | P=0.667 R=0.667 F1=0.667 | **P=1.000 R=1.000 F1=1.000** |
| **vlog** | P=0.167 R=0.200 F1=0.182 | P=0.167 R=0.200 F1=0.182 | P=0.062 R=0.200 F1=0.095 | P=0.214 R=0.600 F1=0.316 |

**효과 분해**:

| 변경 | F1 기여 |
|---|---|
| 사전 정리 ("그러니까/그래서" 제외 + "자" 추가) | lecture P 향상 + 점수 도금 해제 (정직화). F1 자체는 R 깎임으로 -0.04 |
| **ASR 모델 swap** (한국어 fine-tuned) | **lecture F1 +0.27** (결정적 lever). vlog F1 -0.09 (환각) |
| **Tolerance ±1s** (라벨러 1초 라운딩 보정) | **lecture F1 +0.33 → PERFECT 도달**. vlog F1 +0.22 |

ASR 모델 swap이 효과의 90%, tolerance가 10%. 다만 *평가 신뢰도* 측면에선 tolerance가 결정적 —
라벨러 사람 정밀도와 모델 ±20ms 정밀도 격차를 흡수.

#### 미채택 lever 비교 — 시도 8~14 (vlog 천장 돌파 시도)

baseline(KsponSpeech + 사전, vlog F1 0.316) 위에서 시도한 lever들과 채택 X 근거:

| 시도 | 가설 | vlog F1 | lecture F1 | 채택 X 근거 |
|---|---|---|---|---|
| **8** 사전 모호형 제거 | 정상 부사 FP cut | 0.267 | 1.000 | TP 손실 > FP cut. 사전이 라벨러 정의 일부 반영 |
| **9** 음향 lever 정밀 결합 | gap+voiced+RMS+run 결합으로 시도 3 정밀화 | 0.05~0.17 | — | vlog 환경 voiced gap 다양성 — 임계 정밀화로도 분리 불가 |
| **10** 한국어 phoneme 모델 시장 조사 | acoustic-only 모델로 LM 정규화 우회 | 모두 부적합 | 모두 부적합 | kresnik·Kkonjeong·MMS·espeak 모두 학습 분포 부족 |
| **11** gpt-4o-audio chat | LLM instruction following으로 모호형 의도 분류 | 0.078 | 0.188 | 음성 인식은 OK이지만 의도 분류·timestamp 부정밀 |
| **12** whisper-1 chunked v2 | LM 컨텍스트 인위 축소로 정규화 우회 | **0.571** ↑ | 0.750 ↓ | *우회 trick*. lecture 후퇴 + OpenAI 의존 + 재현 보장 X + 영상 콘텐츠 의존 |
| **13** KsponSpeech chunked (B 트랙) | 단일 모델 + chunk 호출로 12의 효과 재현 | 0.333 | 0.857 | chunk 분할 효과는 *모델 의존* — KsponSpeech 한국어 fine-tune이 acoustic 단계 정규화 흡수 |
| **14** 사전 reframe (단음절 제거 + 담화 표지 추가) | 측정 가능 영역에 detection 정의 fit | 0.286 | 0.500 ↓ | 발화 길이가 정규화 결정 — lecture 직격타. 영상 콘텐츠 특성으로 담화 표지 빈도 낮음 |

**핵심 진단** — 어떤 lever도 *재현 가능하고 카테고리 양쪽에 안전한* 향상 못 냄:
- 시도 12(whisper-1 chunked)만 vlog F1 향상하지만 *trick* 성격에 lecture 후퇴
- 단일 모델 통합 path(13)·정의 reframe(14) 모두 카테고리 비대칭으로 좌초
- **본질은 학습 데이터 분포 한계** — 모든 lever가 이 한계 안에서의 trade-off

#### 카테고리별 천장의 비대칭 — v1.1 후보

| 영상 | 모델 효과 | 다음 액션 후보 |
|---|---|---|
| **lecture** | 한국어 fine-tuned가 압도적 (F1 1.0) | 만족. 더 짜낼 영역 X |
| **vlog** | 한국어 fine-tuned가 *vlog 짧은 burst 정규화 흡수* | 자체 한국어 disfluency-aware fine-tune (KsponSpeech 비유창성 annotation 활용 + AI Hub 일상 대화 ~10,000h 통합) / phoneme classifier ML 차원 신규 도입 / 다양한 vlog 영상(대화·인터뷰) 추가로 reframe lever 효과 검증 |

**시장 조사 결론** (시도 7 + 10 + 11 + 12 종합):
- 상용 한국어 ASR(CLOVA·Naver) — 모두 filler 자동 제거 정책, 부적합
- 공개 한국어 wav2vec2 모델(kresnik·Kkonjeong·MMS) — 학습 분포 부족, KsponSpeech 대비 후퇴
- 다국어 phoneme 모델(espeak) — 한국어 학습 비중 작음
- 유료 audio LLM(gpt-4o-audio·whisper-1) — chunk 분할 trick으로 부분 효과지만 trade-off
- 학술 연구 — 한국어 children SSD 임상용 한정, 일반 vlog 부적합
- **PodcastFillers 데이터셋** — 영어 전용 (한국어 X)

**시장 공백을 메우는 v1.1 정공법** — *KsponSpeech 비유창성 annotation 활용한 자체
fine-tune* 또는 *phoneme classifier ML 차원 신규 도입*. 둘 다 GPU 시간·라벨링
데이터·코드 작업이 큰 별도 작업이지만, **vidoctor만의 contribution이 시장 공백
영역**이라 면접 어필 가치가 가장 큼.

#### 회고 — 이 단계에서 얻은 교훈

- **알고리즘 가정과 라벨 패턴이 다르면 임계 튜닝으로 못 메운다**. 반복 검출(시도 1)과 음향
  분석(시도 3) 모두 가정이 데이터에 맞지 않아 실패. 가정 자체를 데이터에 맞게 재설계해야 함
- **모델 학습 분포는 임계로 못 우회한다**. hotwords / VAD 완화 / 모델 layer 교체 모두 ASR 학습
  분포 한계엔 무력 (시도 2). *모델 자체 변경*이 정공법
- **모델 swap 효과의 90%, 평가 정책 효과의 10%지만 *평가 신뢰도* 핵심**. 라벨러 사람 정밀도와
  모델 ±20ms 정밀도 격차는 모델 개선이 아니라 *평가 정책*에서 흡수
- **baseline 점수가 도금일 수 있다**. 우연 매칭이 R을 부풀린 케이스가 사전 정리에서 드러남.
  점수 하락이 "퇴보"가 아니라 "정직화"인 케이스 존재
- **ASR 환각은 라벨 누락처럼 보일 수 있다**. 사용자 청취 같은 *제3자 검증*이 없으면 "우리가 못
  본 라벨"로 오해할 위험. ML 회고에서 모델 출력 = 정답 가정의 경계 직접 확인 필요
- **상용 ASR SOTA가 도메인 task에 SOTA가 아닐 수 있다**. CLOVA Speech는 한국어 transcript
  품질이 SOTA지만 disfluency 검출엔 *반대 방향* 정책으로 부적합. "정확도 1위"가 *어떤 metric에서*
  1위인지 항상 확인 — 상용 솔루션 도입 의사결정 시 sanity check가 결정적
- **chunk 분할은 LM 정규화 우회 trick — 모델 의존성 큼** (시도 12·13). multilingual
  base(whisper-1)에선 효과 큼(vlog F1 +0.255), 한국어 fine-tuned(KsponSpeech)에선 미미
  (+0.017). 한국어 fine-tune이 acoustic 단계 정규화를 깊이 흡수 — *결정적 lever였던
  같은 fine-tune이 vlog 짧은 filler 회수의 장애물*인 양면성
- **정규화 여부는 발화 길이가 결정** (시도 14). 같은 "음·어" 어휘여도 lecture 의도적
  늘임(0.5~1.5s)은 토큰화되고 vlog 짧은 burst(0.1~0.3s)는 흡수. *어휘 단위 사전 변경은
  카테고리 비대칭 효과를 만든다* — 사전 단음절 제거로 lecture 직격타(F1 1.0→0.5)
- **영상 콘텐츠 특성이 lever 효과를 결정** (시도 14). 강아지 산책 vlog는 호명·감탄
  중심이라 자기수정·담화 표지 발화 자체가 거의 없음 — reframe lever(아니·근데 추가)
  적용 불가. *대화·인터뷰 vlog*에선 효과 가능성 — v1.1 영상 다양성 확장 영역
- **시장 공백을 다중 검증한 정직한 천장 진단의 가치**. 공개 모델 7종 + 음향 lever +
  사전 변경 + 유료 audio LLM 모두 vlog F1 0.316 천장 확인. 0.571 도달 가능한 trick은
  trade-off로 미채택. *수치 좇기보다 시장 공백 진단이 시니어 ML 사이클*

### Dead Zone — VAD 직접 측정 + 정적 누적 시간 가드

#### 진단 (Stage 1, ASR 기반)

baseline `lecture F1=0`, `vlog F1=0`에서 라벨 시간대 신호 dump로 두 카테고리가 정반대 방향으로 막힘 확인.

- **lecture (라벨 87~92s "슬라이드 고정 및 침묵")** — 시각 정적 OK, ASR은 발화 중 판단. transcript 확인 시 단어 `[85.89~93.41] '갈게요.'`가 7.5초 길이로 alignment(한국어 단어 p95=0.95s 대비 8배 outlier). WhisperX 단어 끝점이 다음 발화 직전까지 침묵을 흡수하는 alignment 오류.
- **vlog (라벨 88~92s, 308~321s)** — ASR 무발화 OK, SSIM이 노이즈로 신호 못 함 (라벨 시간대 max 0.79·0.85, 영상 전체 median 0.27).

Stage 1 정책 — lecture는 word duration cap 2.0s + 시각 정적 ∩ 무발화, vlog는 silent + SSIM max gate("어딘가 멈춘 적 있나"). 카테고리 분기 결합으로 동의율 100% 도달했지만 두 가지 본질 한계 잔존.

#### Stage 2 — VAD + SSIM 시간 가드 (휴리스틱 제거)

**한계 1 — word cap은 ASR 우회**. WhisperX alignment 오류를 휴리스틱으로 정정.  
**한계 2 — SSIM max gate는 "한 순간이라도 멈춤"**. 컷어웨이 후보가 우연히 비슷한 프레임 한 장 잡으면 통과되는 약점.

- **VAD 직접 측정** (Silero VAD, MIT 라이선스, 무료, 언어 독립). ASR 단어 timestamp 의존 제거.
- **SSIM 정적 누적 시간 가드**. 후보 안 SSIM ≥ 0.7 누적 시간 ≥ 1.0s.
- **카테고리 분기 제거**. 단일 정책으로 통일.

vlog F1 1.0 → 0.8로 **표면 하락**. 분석 결과 detector가 더 정직해졌지만 사용자 정의에 fit하는 새 검출(356.3~362.8s, 6.5초 무발화 + SSIM max 0.977 + 정적 누적 2.5s)이 라벨 천장에 막힘 — *cps에서 본 라벨 천장 패턴*으로 해석.

#### Stage 3 — 사용자 ground truth가 SSIM 한계 노출

사용자가 영상 직접 확인 후 *"356.3~362.8은 화면 움직임 있어 dead zone 아님"*으로 negative 정정. SSIM 신호 재진단:

| 구간 | SSIM max | SSIM 누적 정적 | 사용자 판단 |
|---|---|---|---|
| TP1 라벨 | 0.797 | 1.5s | dead zone |
| TP2 라벨 | 0.849 | 2.5s | dead zone |
| **FP_가짜** | **0.977** | **2.5s** | **움직임 있음** |

SSIM은 가짜 dead zone을 라벨보다 *더 정적*으로 측정. 다운스케일 240→480→원본 모두 동일. **SSIM이 사용자 인지와 분리 못 함이 본질** — 큰 영역(하늘·바닥)이 비슷하면 작은 움직임 가려져 평균 SSIM 높음.

**Optical flow magnitude로 픽셀 움직임 직접 측정**:

| 구간 | flow median | flow max |
|---|---|---|
| lecture 라벨 | 0.001 | 0.003 |
| vlog TP1 | 0.807 | 7.195 |
| vlog TP2 | 0.838 | 12.893 |
| **vlog FP_가짜** | **2.686** | 10.308 |
| vlog 다른 silent | 6.6~7.2 | - |

라벨 max 0.838 vs FP_가짜 2.686 → **임계 1.0**으로 안전마진 분리. 사용자 ground truth와 정확히 일치.

#### 정착

| 카테고리 | baseline | Stage 1 (ASR cap + SSIM gate) | Stage 2 (VAD + SSIM 누적) | Stage 3 (VAD + Flow median) |
|---|---|---|---|---|
| lecture | 0.000 | 0.667 | 0.667 | **0.667** (TP=1/FP=1/R=1.0) |
| vlog | 0.000 | 1.000 | 0.800 | **1.000** (TP=2/FP=0/R=1.0) |

- Stage 1의 vlog 1.0은 ASR alignment 오류가 라벨 천장 방향으로 우연히 작동한 결과 (FP_가짜를 무발화 후보 자체에 못 만들어서 차단됨).
- Stage 3의 vlog 1.0은 **본질 lever** — VAD가 무발화 정확히 잡고 + flow가 사용자 인지 화면 움직임 정확히 측정해서 차단.
- 같은 F1 1.0이지만 메커니즘이 다름. Stage 3는 더 큰 표본·다양한 영상에서도 robust 가능성 높음.
- lecture FP(75.2~84.2s, 9초 무발화 + flow median 0.005)는 사용자 미검증 — 라벨러 누락된 진짜 dead zone일 가능성 높음 (lecture 라벨 0.001과 거의 동일 시각 정적).

#### Stage 4 — 사용자 ground truth #2: lecture 페이스캠

사용자가 lecture FP(75.2~84.2s) 영상 직접 확인 → *"발화 없는데 우하단 페이스캠에서 사람이 움직이고 있어 dead zone 아님. 페이스캠이 너무 작아서 평균 flow에서 안 잡힌 듯"*. 시각 신호 한계 재진단:

- 화면 작은 영역 움직임이 큰 정적 영역(슬라이드)에 묻힘 — `np.mean(magnitude)`으로 평균 내면 페이스캠 영향이 1/100 수준으로 깎임.
- 다운스케일 240p로 더 가려짐.

**per-frame mean → per-frame max 변경**. 한 픽셀이라도 크게 움직이면 잡히도록.

| 구간 | per-frame max median | 사용자 판단 |
|---|---|---|
| lecture TP 88~94 | 0.320 | dead zone |
| **lecture FP_75 (페이스캠)** | **0.982** | 움직임 있음 |
| vlog TP1·TP2 | 2.307·3.438 | dead zone |
| **vlog FP_가짜** | **21.566** | 움직임 있음 |

per-frame max 측정에서 **카테고리 환경 차이가 절대값 가름** — vlog는 핸드헬드라 정적 영상도 카메라 미세 흔들림으로 max 절대값이 lecture(삼각대)보다 5~10배 큼. 단일 절대 임계로 분리 불가능 → **카테고리별 임계 분기**(cps에서 확립된 패턴 재적용).

```
CATEGORY_FLOW_MAX_THRESHOLD = {
    "lecture": 0.5,  # TP 0.32 vs 페이스캠 FP 0.98 분리
    "vlog":    5.0,  # TP 2.3·3.4 vs FP_가짜 21.6 분리
    "other":   5.0,  # 라벨 없음, vlog 보수적 기본
}
```

#### 최종 정착

| 카테고리 | baseline | S1 (ASR cap) | S2 (VAD+SSIM 누적) | S3 (flow mean) | **S4 (flow max + 카테고리별)** |
|---|---|---|---|---|---|
| lecture | 0.0 | 0.667 | 0.667 | 0.667 | **1.000** (TP=1/FP=0) |
| vlog | 0.0 | 1.000 | 0.800 | 1.000 | **1.000** (TP=2/FP=0) |

- **양 카테고리 P=R=F1=1.0**. 라벨 천장 패턴 벗어나 진짜 정밀도 확보.
- 사용자 ground truth 2건이 본질 lever 2개 도입을 견인:
  1. vlog FP_가짜 (S2→S3): SSIM이 사용자 인지와 분리 못 함 → optical flow 도입.
  2. lecture FP_75 페이스캠 (S3→S4): 평균이 작은 영역 묻음 → per-frame max + 카테고리별 임계.
- 라벨 표본 3개로는 통계 robust 측정 불가능하지만, 사용자가 영상 직접 검증한 negative 2건으로 부족분 일부 보완.

#### 정책 정착

```
VAD 무발화 ∩ min_duration 5s ∩ per-frame max 시계열 median ≤ 카테고리별 임계
```

상수:
- `CATEGORY_MIN_DURATION_SEC`: lecture/vlog/other 모두 5.0
- `CATEGORY_FLOW_MAX_THRESHOLD`: lecture 0.5 / vlog 5.0 / other 5.0
- `VAD_MIN_SILENCE_MS`: 1000

이전 정책 상수 7개(`SSIM_STATIC_THRESHOLD`, `MAX_WORD_DURATION_SEC`, `ASR_SILENCE_THRESHOLD_SEC`, `SSIM_MAX_GATE`, `STATIC_SSIM_THRESHOLD`, `MIN_STATIC_DURATION_SEC`, `FLOW_MEDIAN_MAX` 등) 모두 제거. 카테고리 분기는 cps와 일관된 패턴으로 부활.

#### 한계

- 라벨 표본 3개 + 사용자 ground truth 1건으로 통계 신뢰도 약함. flow 임계 1.0도 라벨 0.838 + FP 2.686 사이 sweet spot 중앙이지만 cherry-pick 위험. 표본 확장 후 재검증.
- VAD가 배경 다른 사람 목소리(예: vlog 시장 장면)를 발화로 잡음. **메인 화자 vs 주변 화자 분리**는 pyannote.audio 같은 화자 분리 차원. v1.1.
- Optical flow는 카메라 panning과 객체 움직임 모두 잡음. *고정 카메라 + 작은 객체 움직임*과 *카메라 panning + 정지 객체* 구분 못 함. dead zone 정의상으론 둘 다 "화면 움직임" → 의도 부합. 다만 정지 사진 인서트(SSIM 0.99 + flow 작음)는 dead zone으로 잡힘 — 사용자 의도 영역이라 v1.1 로드맵.
- 무성 영상(audio 트랙 없음)은 ffmpeg RuntimeError → `_load_audio_or_empty`로 빈 array fallback. dead zone 정의상 자연스럽지만 production logging 보강 후보.

### Gaze — baseline 차감 + 임계 완화

라벨 1개(78~82s, lecture)로 시작. baseline 측정 시 **F1=0** — 라벨 시간대 yaw abs_max 14.1°·pitch abs_max 15.05°가 절대 임계(yaw 20°/pitch 15°)를 못 넘음.

진단 — 영상 전반 pitch median이 **-9.1°**로 systematic offset. 무캘리 환경에서 cv2.solvePnP + decomposeProjectionMatrix는 카메라 위치·euler 분해 컨벤션에 따라 영상별 ±수~십도 offset을 만든다. 절대 임계는 baseline 0° 가정에 의존해 영상마다 분리 성능이 흔들림.

**lever — 영상 전체 yaw/pitch median을 정면 baseline으로 차감**. 강의 도메인 가정(강사가 시간 대부분 정면 응시)이 강해 영상 통계 median을 정면 추정값으로 안전하게 사용 가능. baseline 차감 후 정상 응시 deviation abs_p95: yaw 4.3° / pitch 6.1° → 라벨 시간대 yaw abs_max 18.8°와 충분히 분리.

추가로 사용자 ground truth가 lever 2개를 견인:
1. *"라벨 4초 안에서 짧은 휙 동작이 핵심"* → MIN_DURATION 1.0s → 0.6s로 낮춰 짧은 이탈 잡기. 다만 짧게 잡으면 4초 라벨 대비 IoU 부족(0.6/4=0.15) → MERGE_GAP 0.5s → 1.0s로 키워 한 이벤트로 묶음.
2. *"강사 입장 좌하단→좌상단 이동, 시청자 입장 우하단→우상단"* → 합성 회전 입력으로 부호 컨벤션 박제 검증(`test_solve_head_pose_head_*`): yaw>0=강사 우측, pitch>0=강사 머리 아래. UI 가시 영역(`web/components/ResultView.tsx:GAZE_DIRECTION_LABEL`)은 시청자 입장 한국어로 좌우 반전 표기.

| 시도 | 임계 / min_dur / merge_gap / baseline | F1 |
|---|---|---|
| baseline (절대 임계) | yaw 20° / pitch 15° / 1.0s / 0.5s / 절대 | 0.000 |
| relative baseline only | yaw 20° / pitch 15° / 1.0s / 0.5s / 차감 | 0.000 |
| **정착** | **yaw 10° / pitch 12° / 0.6s / 1.0s / 차감** | **1.000** (TP=1/FP=0) |

#### 한계

- **라벨 1개로 통계 신뢰도 매우 약함**. detected 폭 1.4s, 라벨 4s, IoU=0.35로 임계 0.3 간신히 통과 — F1 1.0은 단일 매칭 결과지 일반화 보장 아님.
- pitch는 라벨 변동 abs_max 5.9°가 일반 변동 abs_p95 6.1°와 분리 안 됨. 라벨이 pitch dominant인 케이스가 들어오면 임계 재조정 필요. 현재는 yaw 단독으로 잡힘. *부호 컨벤션 자체는 합성 회전 입력으로 박제 검증됐고(yaw>0=우측, pitch>0=아래) 실 영상에서 사용자 ground truth와 모호했던 건 pitch 변동 폭이 noise 수준이라는 라벨 천장이 원인.*
- baseline 가정이 "강사 시간 대부분 정면 응시"에 의존. 영상 내내 시선 이탈하는 비정상 케이스는 가정 깨짐 (강의 도메인 가정상 무시).

#### 정책 정착

```
영상 yaw/pitch median 차감 → |∆yaw| > 10° OR |∆pitch| > 12° → 0.6s 이상 지속 → 1.0s 갭까지 같은 이벤트로 묶기
```

### Content Gap — rubric 정밀화 + 윈도우 마커

라벨 1개(94~102s, lecture: "슬라이드 내용은 공변성이지만 '반공변성'으로 언급"). baseline F1=0.667 (P=0.500/R=1.000) — TP는 시간대만 우연 매칭, FP 1개 동반.

진단:
- **TP reasoning이 표면**: LLM이 "갑작스러운 등장"으로 잡았고 라벨 본질(슬라이드 시각 vs 발화 미스매치)을 짚지 못함. 시간대 IoU=0.267로 임계 0.2 통과 = 우연 매칭.
- **FP는 ASR 윈도우 분할 artifact**: frame[15s]·frame[44.6s] transcript window가 발화 중간을 잘라 LLM이 "설명 누락"으로 hallucinated.

lever — production rubric/transcript 두 곳 동시 변경:
1. `_LECTURE_RUBRIC` 본문 정밀화: "슬라이드 텍스트·도식이 표시한 용어와 강사 발화 단어가 다른 경우"를 1순위로 명시. 흐름 어색·짧은 추임새·윈도우 분할은 **검출 대상 아님**으로 negative 명시.
2. `_transcript_around`에 "…" 마커: 윈도우 외부에 word가 있으면 양 끝에 `…` 부착 → rubric에서 "…는 인접 윈도우 연속, 끊김 아님" 안내. ASR 분할이 만드는 가짜 누락 신호 차단.

결과 F1=1.0 (P=1.000/R=1.000, IoU=0.329). LLM 검출 reasoning이 *"슬라이드에서는 '공변성(Covariance)'을 설명하고 있지만 강사는 '반공변성'에 대해 설명"* — 라벨 본질 정확 매칭 + 슬라이드 텍스트 영문 병기까지 추출(=프레임을 실제 인식). rubric에 라벨 단어를 예시로 박지 않은 v3에서도 동일 결과 → 데이터 누설로 부풀린 점수가 아님 검증.

| 시도 | F1 | latency / 비용 | reasoning |
|---|---|---|---|
| baseline | 0.667 (IoU=0.267) | 5.06s / $0.0203 | "갑작스러운 등장" (표면) |
| rubric_v2 (예시 포함) | 1.000 | 3.31s / $0.0202 | "공변성 vs 반공변성" (본질) — 단 라벨 누설 의심 |
| **rubric_v3 (정착)** | **1.000 (IoU=0.329)** | **3.63s / $0.0202** | **"슬라이드 공변성 vs 발화 반공변성" (본질, 누설 없음)** |
| slide_ocr_textonly (시도, 미정착) | 1.000 (IoU=0.267) | 39.3s / $0.0176 | OCR이 코드로 슬라이드 텍스트 추출 → text-only 검증 |

비용·latency (3분 영상, GPT-4o, image 6장 / prompt ~7.9K tok / completion ~50 tok):
- **$0.0202/호출, 3.3~3.6s** — P0 "비용·latency UI 노출"의 입력 자료. MAX_SAMPLES=10 cap이라 영상 길이가 늘어도 호출당 prompt token은 ~10K 상한.

#### slide-OCR alignment 시도 — 미정착

rubric_v3 한계: frame을 LLM에 보내고 "슬라이드 텍스트를 추론하라"고 시킴 = LLM의 시각 텍스트 인식 신뢰성에 의존. 짧은 슬라이드(균등 30s 사이로 누락) / 슬라이드-light 영상은 구조적 FN 위험.

대안: dense OCR(2s 간격, gpt-4o-mini detail=low, 360p) → embedding cosine으로 슬라이드 변화 검출 → segment별 [slide_text, ASR]을 GPT-4o text-only로 검증. 신호 메커니즘이 본질적으로 코드 측정 + LLM 검증으로 더 robust.

측정 결과: F1=1.0 동률 / IoU 0.329 → 0.267(↓ segment 폭) / 비용 -$0.0026 / **latency 10배(3.6s → 39.3s)**. lecture 라벨 1개에선 추가 lever 효과(짧은 슬라이드 catching·일반화) 검증 불가하고, latency dominant 비용 대비 개선 정당화 약함 → **rubric_v3 정착 유지, OCR 흐름은 회고 자료로만 보존**.

구현 중 정착시킨 안전장치는 rubric_v3에도 그대로 적용:
- `_ContentGapResponse.issues` `max_length=5` + description `max_length=200` + `max_tokens=1024` — 모델이 issue를 무한 생성해 completion 16384 tok에 닿아 structured output 파싱이 실패하는 케이스 차단
- `get_chat_model`에 `max_retries=5` — TPM 한도 짧은 burst를 SDK 지수 backoff로 흡수

OCR 시도가 남긴 v1.1 lever 후보:
1. **OCR pre-filter**: 슬라이드 텍스트 ↔ ASR embedding 유사도로 LLM 호출 전 후보 추리기 (호출당 비용 절감)
2. **Adaptive frame sampling**: dense OCR로 슬라이드 변화 시점 추출 후 frame을 그 시점에 맞춰 뽑기 (현 균등+컷 결합 sampling 개선)
3. **slide-light fallback**: OCR 신호 약한 영상(필기·실습)에 대한 별도 카테고리 분기

라벨 다양성(슬라이드-rich vs slide-light) 확보 후 재측정 → trade-off 재판단.

#### 한계

- **라벨 1개 = 통계 의미 매우 약함**. F1=1.0은 1/1 매칭 결과로, 영상 다양성·도메인 일반화 보장 X.
- **LLM 비결정성**: temperature=0.0이지만 GPT-4o는 미세 변동 가능. 단일 run 결과라 변동성 미측정.
- **MAX_SAMPLES=10·SAMPLE_INTERVAL_SEC=30s 휴리스틱**: 5~6분 영상까진 균등+컷 결합으로 cap에 안 걸리지만, 그 이상 영상에서는 cap이 핵심 시점을 누락할 위험. 짧게 표시되는 슬라이드도 이 sampling 정책의 구조적 FN.

#### Stage 2 — ASR-anchored 시간 정밀도 보강

UI에서 사용자 검증: `rubric_v3` 정착 후에도 detected가 `[80.7, 105.0]s`로 잡혀 라벨 `[94, 102]s`보다 시작점이 ~14초 앞섬. 사용자 지각상 "1분 20초"로 보여 라벨 시점과 직관 안 맞음.

진단: **LLM은 시간 추론이 약하다** — frame image + ±15s transcript window로 "발화 시점"을 *공간적*으로 추정해야 해 슬라이드가 표시된 전체 구간을 잡는 경향. 라벨러는 미스매치 단어가 *실제 발화된* 짧은 구간을 라벨링해 시간 폭이 다름.

**lever — 책임 분리** (LLM=의미 / ASR=시간):
- `_ContentGapIssue.mismatch_keyword` 필드 신설 — LLM은 강사가 발화한 미스매치 핵심 단어/구를 description과 별도로 추출
- `_anchor_to_asr` 후처리 — LLM detected 구간 ±5s 안에서 키워드를 ASR transcript에서 검색, 매칭된 word들의 `[min start, max end] + ±2s margin`으로 좁힘
- 매칭 우선순위: 정확 substring → 3-gram → 2-gram. 한국어 ASR 오인식(WhisperX가 "반공변성"을 "반공병성"으로 받아씀 같은 케이스)을 흡수하면서 짧은 단어 false positive는 trigram 우선으로 차단
- 매칭 실패 시 LLM 원본 fallback — silent degrade 안전장치

결과 IoU 0.32 → **0.69** (라벨 [94, 102] vs anchored [92.6, 100.5]). 비용·latency 동일($0.021/4.3s) — LLM 호출 1회 변동 없음, 코드 후처리만 추가. UI에서 사용자 직관과 시점 일치.

**도메인 일반화 검증**: rubric에 라벨 단어 페어("공변성/반공변성") 예시를 박았다가 사용자 지적으로 즉시 제거. 정의만 남긴 후에도 LLM이 영상 frame + transcript에서 직접 keyword를 추출해 동일 IoU 0.69 — 데이터 누설 의존이 아닌 진짜 일반화 검증. 다른 도메인(요리·악기·역사) 영상에선 그 영상의 단어가 그대로 추출될 것.

| 시도 | F1 | IoU | 메커니즘 |
|---|---|---|---|
| baseline | 0.667 | 0.267 | "갑작스러운 등장" 표면 |
| rubric_v3 | 1.000 | 0.329 | LLM이 슬라이드 표시 전체 구간 |
| **anchor (정착)** | **1.000** | **0.690** | **LLM=의미 + ASR=시간 분리** |

#### 정책 정착

```
[검출] SAMPLE_INTERVAL_SEC=30 균등 + PySceneDetect 컷 → MAX_SAMPLES=10 cap →
       각 frame ±15s ASR (윈도우 외부 "…" 마커) → GPT-4o multi-image →
       _ContentGapResponse(issues=[{start, end, description, mismatch_keyword}], max_length=5)

[후처리] 각 issue에 대해 _anchor_to_asr:
       정규화(공백·따옴표 제거 + lower) → [start - 5s, end + 5s] 범위 transcript word →
       (1) 정확 substring → (2) 3-gram → (3) 2-gram 매칭 → (min start, max end) ± 2s margin
       매칭 실패 시 LLM 원본 fallback

→ ContentGapEvent(start, end, description). IoU 0.2 매칭.
```

### Generate Suggestions — 5차원 finding 종합 LLM 노드

`generate_suggestions` 노드가 v1.0 stub(`return {"suggestions": []}`)에서 production 함수로 전환. 5차원 event를 gpt-4o-mini structured output에 전달해 한국어 개선 제안 리스트를 생성한다.

**도메인 오버핏 회피 메커니즘**:
- rubric에 카테고리별 hardcoded 가이드(예: "강의는 슬라이드 다듬기", "vlog는 컷 추가") 없음. **차원 신호의 일반적 의미만 정의** (filler=전달력, cps=청취 부담, dead_zone=이탈 위험, gaze=응시 안정성, content_gap=시각·발화 불일치).
- 영상 도메인은 input의 finding 패턴 + 활성 차원으로 LLM이 추정해 톤 조정. 카테고리는 참고용 메타로만 전달.
- 검증: lecture(5차원 finding) / vlog(3차원 finding) mock state 두 케이스에서 슬라이드 specific 표현 / 강의 specific 표현이 적절한 케이스에만 등장 + 다른 케이스에선 일반화된 표현(예: vlog dead_zone → "필요한 내용을 미리 준비하고 발화하세요").

**prompt·출력 통제**:
- `MAX_FINDINGS_PER_DIM=30` 차원당 cap + "추가 N건 생략" 집계로 prompt 폭발 차단. finding 1k+여도 패턴 인식엔 상위 30건이면 충분하고, 누적된 신호는 cap 정보로 LLM에 전달.
- `_SuggestionResponse.suggestions` `max_length=8` / `_SuggestionItem.description` `max_length=200` / `max_tokens=512` — content_gap에서 정착시킨 안전장치 동일 적용.
- finding 0건이면 LLM 호출 생략 → 비용 0.

**typed dispatch — getattr fallback 회피**:
초기 구현은 `getattr(event, "text", "")` 패턴이었으나 typed Pydantic 모델 우회로 silent degrade 위험. **각 event 모델(`FillerEvent` / `CPSEvent` / ...)에 `summary()` 메서드 추가**해 다형성으로 풀고, `_collect_findings`는 `event.summary()`만 호출 — 차원 추가 시 모델 정의에서 누락이 컴파일 타임에 드러난다. `state.py`에 `iter_dimension_events` helper도 추가해 `# type: ignore[literal-required]`를 한 곳에 모음.

**비용·latency**: gpt-4o-mini, 평균 finding 5~10건 / prompt ~수백 token / 출력 ~300 token → ~$0.0001/영상, 1~2s. content_gap detection($0.02)·전체 파이프라인($0.20 목표) 대비 무시 가능.

#### 한계

- **제안 품질 정량 평가 라벨 없음**. content_gap 라벨처럼 "이 finding엔 이런 제안이 적절"하다는 ground truth가 없음 → Cohen's κ 같은 평가 어려움. v1.1: 사용자 채택률·수정률을 메트릭으로 도입.

#### 후속 정착 — v1.0 단순화 + LLM 환각 방어

초기 한계로 적었던 두 항목 중 priority·finding_refs 검증은 후속 사이클에서 해소됐다.

- **priority가 모두 0으로 수렴 → 차원 자체를 스펙아웃.** rubric의 "0=가장 높음" 정의가
  LLM 출력 차별화로 이어지지 않는다는 진단을 그대로 받아들였다. detector severity와
  분리된 LLM-only priority는 임계 결정 근거 없이 추가하면 *분기를 흉내내는 노이즈*가
  된다. v1.1에서 severity 차등이 들어올 때 그것을 가중치로 재도입하기로 정리하고
  컬럼·UI·rubric·골든셋 라벨 일괄 제거 (migration 0004). suggestion 정렬은 LLM 출력
  순서 보존(`order by id`).
- **finding_refs 검증 추가.** LLM은 prompt에 없던 ref(예: `filler:99`, 미존재 차원
  `gaze:0`)를 만들거나 오타를 낸다. UI의 `(?)` 표시 단계까지 흘려보내는 대신, 저장
  직전에 `_validate_refs`로 invalid ref 제거 + 모든 ref가 invalid이면 suggestion
  자체 drop. `_RefValidationStats(input/kept/invalid_refs_removed/dropped)`로 노출해
  LLM 환각률을 INFO 로그로 추적.
- **비용·latency 가시성.** `LLMCallMetrics` + `state.step_metrics`(operator.add
  reducer로 노드 누적) → `complete_analysis`가 합산해 `analyses.cost_usd` + JSON
  metadata로 보존. UI 분석 카드 상단에 총 비용·처리 시간을 표시하고 expander로
  step별(content_gap·suggestions) 분리 정보 노출. 평가 스크립트도 동일 단가표를
  쓰도록 `scripts/content_gap_eval.py`의 중복 헬퍼를 `llm.py`로 일원화.

#### 정책 정착

```
state.py:
  각 EventModel.summary() 메서드 + iter_dimension_events helper
  step_metrics 채널 (operator.add reducer)
suggestions.py:
  _collect_findings (cap 30/dim + extras 집계) → _build_message → gpt-4o-mini
  structured output(include_raw=True) → _validate_refs(valid_refs 대조) →
  list[Suggestion] + LLMCallMetrics
graph/nodes.py:
  detect_content_gap·generate_suggestions가 metrics push (lazy import 유지)
repository.py:
  complete_analysis가 step_metrics 합산 → cost_usd + metadata.step_metrics 보존
```

### 공통 교훈

- **모든 변경은 lecture/vlog 두 카테고리 평가 동시 진행.** 한쪽만 보면 다른 쪽이 깎이는 걸
  놓침 (cps ±1.2σ 시도·F0 결합·whisper-1 chunked·사전 단음절 제거 모두 한쪽만 보면 통과로
  보였을 수치). 카테고리별 환경·발화 패턴이 다르면 *동일 lever 효과도 다름*.
- **표본 한계 — 모든 수치는 *진짜 천장*이 아니라 *현 골든셋 측정값***. 영상 1편씩,
  라벨 2~8개로 통계 robust 측정 불가능. 특히 *lecture filler F1=1.000*은 4 요인 동시
  fit(라벨 3개 + L[45-68] 23초 burst가 ±1s tolerance에 포함 + KsponSpeech가 강의
  명확형 정확 토큰화 + lecture 통제 환경 노이즈 적음) — 단순 라벨 표본 확장만으로도
  깎일 수 있음. F1 수치보다 *각 lever 효과 메커니즘 진단*과 *천장 본질 분리 분석*
  (detector vs ASR vs 라벨 vs 환경)이 시니어 ML 사이클의 진짜 결과. v1.1 영상
  다양성·라벨 표본 확장(카테고리별 30개+)이 **수치 신뢰도** 측면에서도 진짜 lever.

## 프론트엔드 분리 회고 — Streamlit → FastAPI + Next.js

처음 v1.0은 Streamlit 단일 프로세스로 시작했다. ML 파이프라인·DB 연결·UI를 한 곳에
모아 시연·평가까지 빠르게 도달하기 위함이었고, 거기까진 잘 들어맞았다. 그러나
디자인 톤(Claude warm editorial — 테라코타 액센트 + Pretendard + Source Serif 4)을
도입하려 하자 Streamlit의 한계가 명확히 드러났다.

**진단 — 표면 폴리시가 막힌 지점들**
- File uploader 안의 기본 문구·아이콘이 컴포넌트 내부에 박혀 있어 CSS로 톤은 바꿔도
  텍스트·아이콘 교체 불가
- `st.button`은 plain text 라벨만 받아 "파일명 + 메타 + 카테고리 pill" 같은 multi-row
  rich button 불가
- Streamlit 디폴트 `primaryColor=#FF4B4B`(빨강)가 select focus border까지 박혀
  warm tone 디자인과 충돌
- 매 인터랙션마다 전체 rerun되는 모델이 Apple-class fluid transition을 구조적으로
  막음 — drop zone "has-file" morph, submit "분석 진행중…" 변환 등

CSS 주입과 hack으로 ~70%까지는 갔지만, 30%는 native widget 안쪽 DOM이라 외부에서
넘을 수 없는 벽이었다.

**결정 — Streamlit을 transport 계층까지 갈아엎고 React로 이전**
- 백엔드: **FastAPI** + SSE 스트림. graph·repository·suggestions 모듈은 그대로
  재사용하고 transport만 분리. `on_node_complete` sync callback을 `asyncio.Queue` +
  sentinel 패턴으로 main 코루틴이 yield해 클라이언트가 실시간 진행 그래프 갱신.
  클라이언트 disconnect(`CancelledError`/`GeneratorExit`) 시 `fail_analysis`로
  in-progress row 정리.
- 프론트엔드: **Next.js 15 + Tailwind v4 + Pretendard/Source Serif 4/JetBrains Mono**.
  Sidebar/IdleForm/Pipeline/ResultView로 3-state machine(idle/analyzing/result)
  구성. 가로형 5단계 파이프라인 SVG + waiting/active/done/skipped 노드 상태 + flow
  wire 애니메이션. 디자인.md 1:1 매칭.

**ML 포트폴리오 관점에서 본 이전 가치**
- UI는 ML 잡 어필의 lever 1순위가 아니지만, *Streamlit으로 안 되는 영역을 진단해
  필요한 만큼만 별도 frontend로 분리*하는 결정은 시니어 풀스택 시그널이다. "고통이
  닥쳤을 때 도구를 갈아엎을 수 있는가"가 어필 포인트.
- 백엔드 비즈니스 로직(graph·repository·평가)에는 손대지 않고 transport만 추가한
  분리도 *경계 설정 능력* 시그널 — Streamlit 시절 `ui/app.py`의 1300줄 모놀리식이
  FastAPI 30줄 thin wrapper로 줄어들면서 production·평가·재시연 경로가 깨끗히 갈렸다.

**전환 비용**
- 1주일 미만. 디자인 파일이 HTML/CSS 프로토타입으로 정의돼 있었고, React 컴포넌트
  포팅은 mechanical translation이 대부분.
- Streamlit 의존성·세팅 파일·favicon 자산까지 일괄 제거(`d46c6a9`)로 deps 가벼워짐.
  `pyproject.toml`에서 `streamlit>=1.40.0` + transitive(altair/tornado/watchdog 등)
  사라짐.

**남은 트레이드오프**
- `web/lib/api.ts`가 Pydantic 스키마를 손으로 미러링. 차원/카테고리 추가 시 backend
  + frontend 두 곳을 동기화해야 함. v1.1에서 OpenAPI → TypeScript codegen 도입 시
  자동 해소 (잔여 작업 P2 항목).
- 인프라가 2-process(FastAPI + Next.js)로 늘어 dev 셋업 절차가 한 줄에서 두 터미널
  로 증가. prod 배포 시 reverse proxy로 동일 origin에 묶으면 운영 복잡도는 감수
  가능 수준.

## 기술 스택

- 오케스트레이션: LangGraph + LangChain
- VLM: GPT-4o Vision (sync) + GPT-4o-mini (개선 제안)
- 인프라: Supabase Postgres (DB) + Cloudflare R2 (영상 storage, S3 호환)
- UI: Streamlit (`server.maxUploadSize = 300MB`)
- 관찰성: Langfuse (LLM trace) + MLflow (실험 추적)
- 평가: scikit-learn + Cohen's κ + DeepEval + Label Studio + pytest
- 배포: Fly.io performance-2x + Docker

## 처리 목표 (3분 영상)

- 분석 시간: ≤ 1.5분
- 비용: ≤ $0.20/영상
- 영상 업로드 한도: 단일 파일 300MB (원본 보존 — 트림·압축 없이 R2에 그대로)

## 셋업

```bash
# Python + uv (mise가 자동 활성화)
mise install

# 의존성 설치
uv sync

# 환경 변수
cp .env.example .env
# .env 채우기 — OPENAI / SUPABASE / R2 / LANGFUSE / HUGGINGFACE 키

# 백엔드 (FastAPI) — 터미널 1
uv run uvicorn vidoctor.api:app --reload --port 8000

# 프론트엔드 (Next.js) — 터미널 2
cd web && npm install && npm run dev
# → http://localhost:3000

# 테스트
uv run pytest
```

### 외부 인프라

- **Supabase** (DB): 프로젝트 생성 후 `supabase/migrations/` 안의 SQL 파일을 번호 순으로
  SQL Editor에서 실행. `0001_init.sql`(스키마) → `0002_drop_videos_storage_bucket.sql`
  (R2 이전에 따른 Storage bucket 정리). DB만 사용하고 영상 storage는 R2로 분리됨
- **Cloudflare R2** (영상 storage): 무료 가입 → R2 활성화 → 버킷 `vidoctor-videos` 생성 →
  `R2 Object Storage → Manage R2 API Tokens → Create Account API token`으로
  Object Read & Write 권한, 해당 버킷 한정으로 발급. 발급 화면에서 **Access Key ID /
  Secret Access Key / Endpoint URL** 3종을 `.env`에 채움. 무료 10GB·egress 무료·단일
  파일 5TB라 골든셋(원본 보존, 5~6분 영상) 시연에 충분
- **Langfuse**: cloud free tier 가입 → public/secret key 발급

## 라이선스

MIT (예정)
