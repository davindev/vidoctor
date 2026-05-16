# 평가·스모크 스크립트

`src/vidoctor/`가 프로덕션 파이프라인이라면 `scripts/`는 그 위의 평가·진단 도구. 5개 차원(filler/cps/dead_zone/gaze/content_gap)의 임계값 결정은 모두 여기서 측정된 P/R/F1과 MLflow run 비교에 근거한다.

## 데이터 폴더 (`data/golden/`)

```
data/golden/
├── labels/                  사람이 라벨링한 정답 CSV
│   ├── lecture_labels.csv
│   └── vlog_labels.csv
├── inputs/                  영상 + 재계산 가능한 feature 캐시
│   ├── lecture.mp4, vlog.mp4
│   ├── transcript_*.json
│   ├── f0_*.npz
│   ├── flow_max_*.npz
│   ├── gaze_pose_*.npz
│   └── lecture_t*.png
└── eval_dumps/              평가 결과 JSON (차원별 / 누적)
    ├── filler/  cps/  dead_zone/  gaze/  content_gap/
    └── smoke/
```

## 스크립트

| 파일 | 역할 |
|---|---|
| `filler_eval.py` | 한국어 추임새 검출 평가 |
| `cps_eval.py` | 발화 속도(글자/초) 이상 검출 평가 |
| `dead_zone_eval.py` | 무발화 + 정적 구간 검출 평가 |
| `gaze_eval.py` | 시선 이탈(head pose) 검출 평가 |
| `content_gap_eval.py` | LLM 기반 슬라이드↔발화 미스매치 평가 |
| `smoke_run.py` | 전체 graph 1회 실행 (회귀 가드) |
| `eval_all.py` | 5차원 × 영상 일괄 실행 (subprocess) |
| `mlflow_ui.sh` | 로컬 MLflow UI 띄우기 (`http://127.0.0.1:5001`) |

## CLI 인자

5개 평가 스크립트가 `_script_lib.build_eval_parser`로 공유하는 공통 인자:

| 인자 | 종류 | 설명 |
|---|---|---|
| `video_path` | positional | 평가할 mp4 경로 |
| `labels_csv` | positional | golden 라벨 CSV 경로 |
| `--run-name` | required | MLflow run 이름. 비교 단위 (예: `baseline_lecture`, `stage11_kspon`) |
| `--no-cache` | flag | transcript/feature 캐시 재사용 안 함 |
| `--no-mlflow` | flag | MLflow 기록 생략 (로컬 디버깅) |
| `--force` | flag | 기존 dump JSON 덮어쓰기 허용 |

차원별 추가 인자:

- **`cps_eval`** — `--no-pitch` (F0 multi-feature 비활성)
- **`dead_zone_eval`** — `--category` ∈ {lecture, vlog, other} (required), `--min-duration`, `--flow-threshold`
- **`gaze_eval`** — `--yaw-threshold`, `--pitch-threshold`, `--min-duration`, `--merge-gap`, `--no-baseline`
- **`content_gap_eval`** — `--category` ∈ {lecture, other} (required), `--model` (예: gpt-4o, gpt-4o-mini)

## 튜닝 사이클

```
1. 코드 수정 (예: cps σ 임계 ±1.5 → ±1.3)
        │
        ▼
2. uv run python scripts/cps_eval.py \
     data/golden/inputs/lecture.mp4 \
     data/golden/labels/lecture_labels.csv \
     --run-name stage16_tighter
        │
        ▼
3. 산출물:
   - mlruns/<exp>/<uuid>/                              MLflow run (metric + param)
   - data/golden/eval_dumps/cps/lecture_stage16_tighter.json   detected/labels 본문
        │
        ▼
4. 비교:
   - bash scripts/mlflow_ui.sh → http://127.0.0.1:5001
   - JSON dump diff (어떤 이벤트가 새로/사라졌는지)
        │
        ▼
5. F1 개선되면 commit, 아니면 되돌리기
```

전체 5차원 × 2영상 일괄 실행:
```bash
uv run python scripts/eval_all.py --tag stage16
```

## 캐시 무효화

| 캐시 | 위치 | 무효화 |
|---|---|---|
| transcript | `data/golden/inputs/transcript_*.json` | `--no-cache` |
| F0 (cps) | `data/golden/inputs/f0_*.npz` | `--no-cache` |
| flow (dead_zone) | `data/golden/inputs/flow_max_*.npz` | 파일 삭제 |
| pose (gaze) | `data/golden/inputs/gaze_pose_*.npz` | 파일 삭제 |

WhisperX 모델 교체 시 `transcript_*_{model_tag}.json`이 자동 분리되므로 default vs 한국어 fine-tuned 결과가 안 섞인다.

## 평가 산출물

| 종류 | 위치 | 용도 |
|---|---|---|
| MLflow run | `mlruns/<exp>/<uuid>/` | run간 metric·param 정렬 비교 |
| JSON dump | `data/golden/eval_dumps/<dim>/` | detected/labels 본문 — 사후 디버깅 |
| feature 캐시 | `data/golden/inputs/` | 반복 평가 가속 |

## 설계 포인트

- **공통 인프라 추출** (`_script_lib`): argparse·로깅·dump 가드·MLflow 셋업·캐시 헬퍼를 한 곳에 모아 5개 스크립트의 boilerplate 0.
- **`--run-name` 접두 로깅**: 동시 평가 실행 시 `[stage16_lecture] [scripts.cps_eval] ...` 형식으로 grep 분리 가능.
- **dump 덮어쓰기 가드**: 같은 `--run-name` 재실행 시 옛 detected/labels 본문 보존을 위해 `--force` 없이 abort.
- **모델 태그 기반 캐시 키**: `VIDOCTOR_WHISPER_MODEL` 환경변수가 캐시 파일명에 박혀 모델 비교 실험에서 결과 오염 방지.
- **MLflow 실패 격리**: 평가 결과 JSON dump가 끝난 뒤 MLflow 호출이 실패해도 warning만 남기고 계속 진행.
