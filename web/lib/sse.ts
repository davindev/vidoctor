/** POST /api/analyze SSE 스트림 컨슈머.
 *
 * 표준 EventSource는 GET·동일 origin·credentials만 받아 multipart 업로드와 안 맞음.
 * fetch + ReadableStream 으로 직접 SSE 프레임을 파싱한다 — `event: <name>\ndata: <json>\n\n`. */

import { API_BASE, assertOk, type Category, type CategoryChoice } from "./api";

/** 클라이언트 측 진행 단계. 서버는 "downloading" | "classifying" | "uploading"을 보내고,
 * "running"은 `uploaded` 이벤트 후 클라이언트가 derive. */
export type AnalyzingPhase =
  | "downloading"
  | "classifying"
  | "uploading"
  | "running";

export type AnalyzeEvent =
  | { type: "status"; phase: "downloading" | "classifying" | "uploading" }
  | { type: "metadata"; filename: string }
  | { type: "category"; category: Category }
  | { type: "started"; analysis_id: string }
  | { type: "uploaded" }
  | { type: "node"; name: string }
  | { type: "complete"; analysis_id: string }
  | { type: "error"; message: string; analysis_id: string | null };

/** 입력 소스 — 파일 업로드 또는 유튜브 URL. */
export type AnalyzeSource =
  | { kind: "file"; file: File }
  | { kind: "url"; url: string };

export interface AnalyzeOptions {
  source: AnalyzeSource;
  category: CategoryChoice;
  signal?: AbortSignal;
  onEvent: (ev: AnalyzeEvent) => void;
}

/** SSE 라인을 누적하며 빈 줄(=`\n\n`)에서 한 이벤트를 분리. */
function parseFrame(frame: string): AnalyzeEvent | null {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    const payload = JSON.parse(data) as Record<string, unknown>;
    return { type: event, ...payload } as AnalyzeEvent;
  } catch {
    return null;
  }
}

export async function postAnalyze(opts: AnalyzeOptions): Promise<void> {
  const form = new FormData();
  form.append("category", opts.category);
  if (opts.source.kind === "file") {
    form.append("file", opts.source.file);
  } else {
    form.append("url", opts.source.url);
  }

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: form,
    signal: opts.signal,
  });
  await assertOk(res);
  if (!res.body) {
    throw new Error("응답 본문이 비어 있습니다 (SSE 스트림 없음)");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // SSE 프레임 구분자 `\n\n` 기준으로 누적 버퍼를 쪼개서 처리.
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = parseFrame(frame);
      if (ev) opts.onEvent(ev);
    }
  }
}
