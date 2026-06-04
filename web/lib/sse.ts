/** POST /api/analyze — 분석을 시작하고 analysis_id를 받는다.
 *
 * 분석은 서버(Modal)가 클라이언트 연결과 무관하게 끝까지 실행하므로, 진행 상황은
 * 더 이상 스트리밍하지 않고 GET /api/analyses/{id}/status 폴링으로 추적한다. */

import { API_BASE, assertOk, type CategoryChoice } from "./api";

/** 클라이언트 측 진행 단계. 사전 단계(다운로드·분류·업로드)는 POST 응답 전까지만
 * 표시하고, POST가 analysis_id를 반환하면 "running"으로 고정해 폴링으로 넘어간다. */
export type AnalyzingPhase =
  | "downloading"
  | "classifying"
  | "uploading"
  | "running";

/** 입력 소스 — 파일 업로드 또는 유튜브 URL. */
export type AnalyzeSource =
  | { kind: "file"; file: File }
  | { kind: "url"; url: string };

export interface AnalyzeOptions {
  source: AnalyzeSource;
  category: CategoryChoice;
  signal?: AbortSignal;
}

/** 분석을 시작하고 analysis_id를 반환한다. 업로드·분류·Modal 위임까지만 기다리며,
 * 분석 진행은 호출자가 fetchStatus 폴링으로 추적한다. */
export async function postAnalyze(opts: AnalyzeOptions): Promise<string> {
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
  const data = (await res.json()) as { analysis_id: string };
  return data.analysis_id;
}
