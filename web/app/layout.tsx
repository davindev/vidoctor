import type { Metadata } from "next";
import { JetBrains_Mono, Source_Serif_4 } from "next/font/google";
import "./globals.css";

// next/font가 self-host로 폰트를 다운로드해 빌드에 묶어주고, CSS 변수를 root에 주입.
// globals.css의 @theme `--font-serif`/`--font-mono`가 이 변수를 참조하도록 연결.
const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  variable: "--font-source-serif-4",
  weight: ["400", "500", "600"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Vidoctor — 비디오 분석",
  description: "AI 영상 감수 에이전트 — 5차원 분석 + 개선 제안",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="ko"
      className={`${sourceSerif.variable} ${jetbrainsMono.variable}`}
    >
      <body>{children}</body>
    </html>
  );
}
