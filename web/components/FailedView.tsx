"use client";

import type { Category } from "@/lib/api";
import { ResultHeader, ResultPage } from "./ResultHeader";

interface Props {
  category: Category | null;
  filename: string | null;
  errorMessage: string;
  onNewAnalysis: () => void;
}

/** 분석 실패 전용 화면 — 진행률 UI 없이 실패 사유와 다음 행동(새 분석)만 보여준다. */
export function FailedView({
  category,
  filename,
  errorMessage,
  onNewAnalysis,
}: Props) {
  return (
    <ResultPage>
      <ResultHeader trailing="실패" filename={filename} category={category} />
      <section className="rounded-[14px] border border-[#EAC8C2] bg-[#FBEFEB] px-8 py-12 text-center">
        <div className="mx-auto grid h-11 w-11 place-items-center rounded-full bg-[#F2D5CD] text-[20px] font-semibold text-danger">
          !
        </div>
        <h2 className="mt-4 text-[16px] font-semibold tracking-[-0.01em] text-[#6E3A33]">
          분석을 완료하지 못했습니다
        </h2>
        <p className="mx-auto mt-2 max-w-[460px] text-[13.5px] leading-[1.65] text-[#8A5249]">
          {errorMessage}
        </p>
        <button
          type="button"
          onClick={onNewAnalysis}
          className="mt-6 rounded-full bg-accent px-5 py-2.5 text-[13px] font-medium text-white transition-[background] duration-150 hover:bg-[#C26344]"
        >
          새 영상 분석
        </button>
      </section>
    </ResultPage>
  );
}
