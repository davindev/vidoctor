"use client";

import type { Category } from "@/lib/api";
import { Pipeline } from "./Pipeline";
import { ResultHeader, ResultPage } from "./ResultHeader";

interface Props {
  category: Category;
  filename: string | null;
  uploadDone: boolean;
  completed: Set<string>;
  errorMessage: string | null;
}

export function AnalyzingView({
  category,
  filename,
  uploadDone,
  completed,
  errorMessage,
}: Props) {
  return (
    <ResultPage>
      <ResultHeader trailing="진행 중" filename={filename} category={category} />
      <Pipeline
        category={category}
        uploadDone={uploadDone}
        completed={completed}
        errorMessage={errorMessage}
      />
    </ResultPage>
  );
}
