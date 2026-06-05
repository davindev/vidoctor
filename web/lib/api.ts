/** Vidoctor FastAPI 클라이언트 — Pydantic 스키마와 1:1 매핑되는 타입 + REST 헬퍼.
 *
 * 분석 시작(`POST /api/analyze`)은 별도 `lib/analyze.ts`에서 처리. */

export type Category = "lecture" | "vlog" | "other";

/** 사용자가 폼에서 고르는 값. "auto"는 백엔드 분류기에 위임한다. */
export type CategoryChoice = Category | "auto";

export type Dimension =
  | "filler"
  | "cps"
  | "dead_zone"
  | "gaze"
  | "content_gap";

export const CATEGORY_LABEL: Record<Category, string> = {
  lecture: "강의",
  vlog: "브이로그",
  other: "기타",
};

/** 폼 select 옵션. 순서는 표시 순서와 일치 (auto가 첫번째). */
export const CATEGORY_CHOICE_LABEL: Record<CategoryChoice, string> = {
  auto: "자동 분류",
  lecture: CATEGORY_LABEL.lecture,
  vlog: CATEGORY_LABEL.vlog,
  other: CATEGORY_LABEL.other,
};

export const DIMENSION_LABEL: Record<Dimension, string> = {
  filler: "추임새",
  cps: "말 속도",
  dead_zone: "정적 구간",
  gaze: "시선 이탈",
  content_gap: "내용 불일치",
};

/** state.py CATEGORY_DIMENSIONS 매핑을 그대로 옮긴 값. graph 노드 활성 여부 derive에 사용. */
export const CATEGORY_DIMENSIONS: Record<Category, Dimension[]> = {
  lecture: ["filler", "cps", "dead_zone", "gaze", "content_gap"],
  vlog: ["filler", "cps", "dead_zone"],
  other: ["filler", "cps", "dead_zone", "content_gap"],
};

export const DIMENSION_ORDER: Dimension[] = [
  "filler",
  "cps",
  "dead_zone",
  "gaze",
  "content_gap",
];

export interface AnalysisListItem {
  id: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  category: Category | null;
  storage_path: string | null;
  status: string | null;
  filename: string | null;
}

/** 진행 상태 폴링 응답 — videos.status + 노드 진행률. */
export interface AnalysisStatus {
  status: "analyzing" | "completed" | "failed" | null;
  progress: { completed_nodes?: string[]; phase?: string };
  error: string | null;
}

export interface FindingItem {
  dimension: Dimension;
  start: number;
  end: number;
  payload: Record<string, unknown>;
}

export interface SuggestionItem {
  text: string;
  finding_refs: string[];
}

export interface StepMetric {
  step: string;
  model: string;
  cost_usd: number;
  latency_sec: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface SpeakerTurn {
  start: number;
  end: number;
  speaker: string;
  word_count: number;
  text_preview: string;
}

export interface SpeakerDiarization {
  main_speaker: string;
  durations: Record<string, number>;
  turns: SpeakerTurn[];
}

export interface AnalysisDetail {
  id: string;
  started_at: string | null;
  finished_at: string | null;
  cost_usd: number | null;
  category: Category | null;
  storage_path: string | null;
  duration_sec: number | null;
  filename: string | null;
  findings: Record<Dimension, FindingItem[]>;
  suggestions: SuggestionItem[];
  step_metrics: StepMetric[];
  speaker_diarization: SpeakerDiarization | null;
}

/** 5차원 색상 팔레트 — Analysis.html 디자인 토큰. issue table 좌측 dot + ts-time 색상. */
export const DIMENSION_COLOR: Record<Dimension, string> = {
  filler: "#B97A3D",
  cps: "#6E7C45",
  dead_zone: "#6B5F58",
  gaze: "#8E5A8C",
  content_gap: "#B5483D",
};

export const DIMENSION_DESC: Record<Dimension, string> = {
  filler: '"음", "어", "이제" 같은 군더더기 말이 자주 나온 구간',
  cps: "평소 말 속도와 비교해 유난히 빠르거나 느린 구간",
  dead_zone: "5초 넘게 아무 말도 없고 화면도 거의 멈춰 있는 구간",
  gaze: "카메라를 정면으로 보지 않고 시선이 다른 곳으로 새는 구간",
  content_gap: "화면에 보이는 내용과 실제 말하는 내용이 어긋나는 구간",
};

export interface VideoUrlResponse {
  url: string | null;
}

/** API base URL — dev에선 `NEXT_PUBLIC_API_BASE=http://localhost:8000`처럼 FastAPI를
 * 직접 가리키고 (Next dev proxy 우회로 multipart streaming의 ECONNRESET 회피), prod
 * 동일 origin 배포에선 미설정 → 빈 문자열로 떨어져 relative path가 그대로 동작. */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

const RATE_LIMIT_FALLBACK =
  "요청이 너무 많습니다. 잠시 후 다시 시도해주세요.";

const UNIT_KO: Record<string, string> = {
  second: "초",
  minute: "분",
  hour: "시간",
  day: "일",
};

/** slowapi 본문의 "N per M unit" 패턴(예: "3 per 1 hour", "5 per 10 minutes")을
 * 한국어로 변환. multiplier가 1이면 생략("시간당"), 그 외엔 명시("10분당").
 * 매칭 실패 시엔 영문 leak 방지를 위해 한글 fallback으로 떨어진다. */
function formatRateLimit(raw: string): string {
  const m = raw.match(/(\d+)\s*per\s*(\d+)?\s*(second|minute|hour|day)s?/i);
  if (!m) return RATE_LIMIT_FALLBACK;
  const [, count, multiplierRaw, unit] = m;
  const multiplier = multiplierRaw ? Number(multiplierRaw) : 1;
  const unitKo = UNIT_KO[unit.toLowerCase()] ?? unit;
  const period = multiplier === 1 ? `${unitKo}당` : `${multiplier}${unitKo}당`;
  return `요청 한도(${period} ${count}건)에 도달했습니다. 잠시 후 다시 시도해주세요.`;
}

/** 백엔드 응답 본문에서 표시용 메시지를 뽑아낸다.
 * FastAPI `{detail}` / slowapi `{error}` 양쪽을 지원. */
function extractDetail(body: string): string | null {
  try {
    const j = JSON.parse(body) as { detail?: unknown; error?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (typeof j.error === "string") return j.error;
  } catch {
    /* 본문이 JSON이 아니면 null로 폴백 */
  }
  return null;
}

/** HTTP 응답이 실패면 사용자 친화적 한국어 메시지로 Error 던지기. fetch 호출자가 공유.
 * 원본 status·body는 console.error + Error.cause로 보존해 디버깅 컨텍스트를 잃지 않는다. */
export async function assertOk(res: Response): Promise<void> {
  if (res.ok) return;
  const body = await res.text().catch(() => "");
  const detail = extractDetail(body);

  let message: string;
  if (res.status === 429) {
    message = detail ? formatRateLimit(detail) : RATE_LIMIT_FALLBACK;
  } else if (res.status === 404) {
    message = detail ?? "요청한 리소스를 찾을 수 없습니다.";
  } else if (res.status === 413) {
    message = detail ?? "파일 크기가 서버 허용 한도를 초과했습니다.";
  } else if (res.status === 400) {
    message = detail ?? "요청 형식이 올바르지 않습니다.";
  } else if (res.status >= 500) {
    message = detail ?? "서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.";
  } else {
    message = detail ?? "요청을 처리하지 못했습니다.";
  }

  console.error(`[api] ${res.status} ${res.statusText} ${res.url}`, body);
  throw new Error(message, { cause: { status: res.status, statusText: res.statusText, body } });
}

// Fly cold-start 구간엔 프록시가 CORS 헤더 없는 응답을 내 fetch가 실패하므로 재시도.
const RETRY_DELAYS_MS = [600, 1500, 3000];

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

/** 4xx는 재시도해도 결과가 같아 제외, 네트워크 실패·5xx만 재시도. */
function isRetriable(error: unknown): boolean {
  if (
    error instanceof Error &&
    typeof error.cause === "object" &&
    error.cause !== null &&
    "status" in error.cause
  ) {
    return (error.cause as { status: number }).status >= 500;
  }
  // fetch 네트워크 실패는 cause 없는 TypeError.
  return error instanceof TypeError;
}

async function getJSON<T>(path: string): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(`${API_BASE}${path}`);
      await assertOk(res);
      return (await res.json()) as T;
    } catch (error) {
      if (attempt >= RETRY_DELAYS_MS.length || !isRetriable(error)) throw error;
      await sleep(RETRY_DELAYS_MS[attempt]);
    }
  }
}

export async function fetchAnalyses(): Promise<AnalysisListItem[]> {
  return getJSON<AnalysisListItem[]>("/api/analyses?limit=20");
}

/** 진행 상태 경량 폴링 — 분석 중 3초 간격으로 호출. */
export async function fetchStatus(id: string): Promise<AnalysisStatus> {
  return getJSON<AnalysisStatus>(
    `/api/analyses/${encodeURIComponent(id)}/status`,
  );
}

export async function fetchAnalysis(id: string): Promise<AnalysisDetail> {
  return getJSON<AnalysisDetail>(`/api/analyses/${encodeURIComponent(id)}`);
}

export async function fetchVideoUrl(id: string): Promise<string | null> {
  const r = await getJSON<VideoUrlResponse>(
    `/api/analyses/${encodeURIComponent(id)}/video-url`,
  );
  return r.url;
}

export async function deleteAnalysis(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/analyses/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  await assertOk(res);
}
