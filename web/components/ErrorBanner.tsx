/** Claude warm editorial 톤의 danger 배너. 5곳에서 같은 빨간 박스를 인라인했었음.
 *
 * variants:
 * - default: 한 줄 경고 (Dropzone 검증, UrlInput invalid, Pipeline 분석 실패, ResultView 삭제 실패)
 * - titled: 제목 + 본문 (IdleForm lastError)
 *
 * `icon=false`로 끄면 SVG 없이 텍스트만 — Pipeline·ResultView처럼 prefix("분석 실패:")가
 * 메시지에 묶여 있는 경우 사용. */

import type { ReactNode } from "react";

interface Props {
  title?: string;
  message: ReactNode;
  icon?: boolean;
  className?: string;
}

export function ErrorBanner({ title, message, icon = true, className }: Props) {
  return (
    <div
      className={`flex items-start gap-2.5 rounded-lg border border-danger-soft bg-danger-tint px-3 py-2.5 text-[13px] text-danger ${
        className ?? ""
      }`}
    >
      {icon && (
        <svg
          className="mt-[2px] flex-shrink-0"
          width="14"
          height="14"
          viewBox="0 0 14 14"
          fill="none"
        >
          <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.4" />
          <path
            d="M7 4V7.5M7 9.5V10"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        </svg>
      )}
      <div className="leading-[1.55]">
        {title && <div className="font-semibold">{title}</div>}
        <div className={title ? "mt-0.5 text-ink-2" : ""}>{message}</div>
      </div>
    </div>
  );
}
