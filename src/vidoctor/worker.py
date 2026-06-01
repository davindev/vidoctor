"""분석 1회 전용 subprocess. 종료 시 OS가 native heap(ctranslate2/MediaPipe/numpy)
까지 회수해 main 프로세스 누적 OOM을 차단.

stdout=JSONL 통신, stderr=fly logs inherit. 종료 코드: 0 성공, 1 실패.
"""

from __future__ import annotations

import os
import sys

# 라이브러리/로그 stdout이 JSONL 채널과 섞이면 main 파싱이 깨짐.
# fd 1을 dup으로 보존하고 그 자리에 stderr 사본을 놓아 모든 print/log를 stderr로.
_protocol_fd = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import argparse  # noqa: E402
import asyncio  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from typing import Any  # noqa: E402

_log = logging.getLogger(__name__)

# line-buffered: main readline 지연 방지.
_protocol = os.fdopen(_protocol_fd, "w", buffering=1)


def _emit(msg: dict[str, Any]) -> None:
    _protocol.write(json.dumps(msg, ensure_ascii=False) + "\n")
    _protocol.flush()


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    # audio_16k(ndarray)는 main에서 안 쓰고 직렬화도 무거워 제외.
    out: dict[str, Any] = {}
    for key, value in state.items():
        if key == "audio_16k":
            continue
        if isinstance(value, list):
            serialized: list[Any] = []
            for item in value:
                if hasattr(item, "model_dump"):
                    serialized.append(item.model_dump())
                elif dataclasses.is_dataclass(item) and not isinstance(item, type):
                    serialized.append(dataclasses.asdict(item))
                else:
                    serialized.append(item)
            out[key] = serialized
        else:
            out[key] = value
    return out


async def _amain() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--analysis-id", required=True)
    args = parser.parse_args()

    from vidoctor.log_setup import analysis_id_var, configure_logging

    configure_logging()
    analysis_id_var.set(args.analysis_id)

    def _on_node(name: str) -> None:
        _emit({"event": "node", "name": name})

    try:
        from vidoctor.graph import run_analysis

        state = await run_analysis(
            args.video_path, args.category, on_node_complete=_on_node
        )
        _emit({"event": "complete", "state": _serialize_state(dict(state))})
        return 0
    except Exception as e:  # noqa: BLE001
        _log.exception("분석 워커 실패")
        from vidoctor.errors import SafeError

        public = (
            e.public_message
            if isinstance(e, SafeError)
            else "분석 중 오류가 발생했습니다."
        )
        _emit({"event": "error", "message": public})
        return 1


def main() -> None:
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
