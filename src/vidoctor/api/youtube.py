"""유튜브 URL → 임시 mp4 파일 다운로드.

`/api/analyze`가 file 대신 url을 받았을 때 분석 파이프라인에 넘길 로컬 mp4를 만든다.
yt-dlp는 1000+ 호스트를 지원하지만 시연 일관성·악용 방지를 위해 youtube.com / youtu.be만
허용한다. 길이 cap(10분)은 사전 메타 조회로 검증해 큰 영상의 트래픽을 차단한다.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import yt_dlp as _yt_dlp

from vidoctor.errors import SafeError

# 모듈 stubs가 일부만 typed라 dict[str, object] params·DownloadError 접근에서 pyright가
# 걸린다. 런타임에는 둘 다 안정적으로 존재 — 모듈 전체를 Any로 캐스팅해 좁히지 않는다.
yt_dlp: Any = cast(Any, _yt_dlp)

_log = logging.getLogger(__name__)

MAX_DURATION_SEC = 600  # 10분 cap — IdleForm 안내 문구와 동기화

# youtu.be/<id>, youtube.com/watch?v=<id>, youtube.com/shorts/<id> 등 허용. music.youtube.com,
# 라이브 스트림 페이지 등은 의도적으로 제외. 정밀 검증은 yt-dlp가 메타 단계에서 한다.
_HOST_PATTERN = re.compile(r"^https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", re.IGNORECASE)


class YouTubeIngestError(SafeError):
    """사용자에게 그대로 노출할 수 있는 다운로드/검증 단계 에러."""


def _is_supported_url(url: str) -> bool:
    return bool(_HOST_PATTERN.match(url.strip()))


def _download_sync(url: str) -> tuple[Path, str]:
    """blocking yt-dlp 호출. (mp4_path, video_title) 반환."""
    if not _is_supported_url(url):
        raise YouTubeIngestError("유튜브 URL만 지원합니다 (youtube.com / youtu.be).")

    tmp = NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    out_path = Path(tmp.name)

    # 단일 YoutubeDL 컨텍스트에서 메타 추출 → 길이 검증 → 다운로드. extract_info(download=False)
    # 후 같은 ydl 인스턴스의 process_ie_result(download=True)로 처리하면 webpage/player JS를
    # 두 번 fetch하지 않는다. noplaylist=True가 없으면 watch?v=...&list=... URL에서 플레이리스트
    # 전체를 펼쳐 fan-out 호출이 일어남.
    opts: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(out_path),
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "overwrites": True,
        "noplaylist": True,
        # youtube가 응답 끊거나 느린 chunk를 보내도 분석이 무한 대기하지 않도록 stuck 가드.
        "socket_timeout": 30,
        # 재시도는 yt-dlp가 알아서. 한 fragment 5회면 가벼운 jitter는 흡수, 영구 fail은 fast.
        "retries": 5,
        "fragment_retries": 5,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise YouTubeIngestError("영상 정보를 가져오지 못했습니다.")

            duration = int(info.get("duration") or 0)
            if duration <= 0:
                raise YouTubeIngestError(
                    "영상 길이를 확인할 수 없어 분석을 시작할 수 없습니다."
                )
            if duration > MAX_DURATION_SEC:
                mins, secs = divmod(duration, 60)
                raise YouTubeIngestError(
                    f"영상이 너무 깁니다. 10분 이내만 지원합니다 (현재 {mins}분 {secs}초)."
                )

            ydl.process_ie_result(info, download=True)
            title = str(info.get("title") or "video")
    except yt_dlp.DownloadError as e:
        out_path.unlink(missing_ok=True)
        raise YouTubeIngestError(f"다운로드 실패: {e}") from e
    except YouTubeIngestError:
        out_path.unlink(missing_ok=True)
        raise

    if not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        raise YouTubeIngestError("다운로드는 끝났지만 파일이 비어 있습니다.")

    return out_path, title


async def download_youtube(url: str) -> tuple[Path, str]:
    """blocking yt-dlp 호출을 스레드풀로 위임."""
    return await asyncio.to_thread(_download_sync, url)
