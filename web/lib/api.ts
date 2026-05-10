/** Vidoctor FastAPI 클라이언트 — Pydantic 스키마와 1:1 매핑되는 타입 + REST 헬퍼.
 *
 * SSE 진행 스트림(`POST /api/analyze`)은 별도 `lib/sse.ts`에서 처리. */

export type Category = "lecture" | "vlog" | "other";

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

/** API base URL — `NEXT_PUBLIC_API_BASE` 미설정 시 빈 문자열로 fall back해 next.config의
 * rewrite proxy를 통하게 한다. dev 환경에서는 직접 FastAPI를 가리켜 proxy 회피 권장. */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchAnalyses(): Promise<AnalysisListItem[]> {
  return getJSON<AnalysisListItem[]>("/api/analyses?limit=20");
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
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
}
