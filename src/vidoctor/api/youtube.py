"""유튜브 URL → 임시 mp4. 시연 일관성·악용 방지로 youtube.com / youtu.be만 허용."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import yt_dlp as _yt_dlp

from vidoctor.config import MAX_VIDEO_DURATION_SEC, MIN_VIDEO_DURATION_SEC
from vidoctor.errors import SafeError

# 모듈 stubs가 일부만 typed라 dict[str, object] params·DownloadError 접근에서 pyright가
# 걸린다. 런타임에는 둘 다 안정적으로 존재 — 모듈 전체를 Any로 캐스팅해 좁히지 않는다.
yt_dlp: Any = cast(Any, _yt_dlp)

# 유튜브는 길이가 메타데이터에 있어 다운로드 전 차단 가능. 파일 업로드와 같은 길이 상·하한 공유.
_MAX_DURATION_SEC = MAX_VIDEO_DURATION_SEC
_MIN_DURATION_SEC = MIN_VIDEO_DURATION_SEC

# music.youtube.com, 라이브 스트림 페이지 등은 의도적 제외. 정밀 검증은 yt-dlp 위임.
_HOST_PATTERN = re.compile(r"^https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", re.IGNORECASE)


class YouTubeIngestError(SafeError):
    """사용자에게 그대로 노출할 수 있는 다운로드/검증 단계 에러."""


def _is_supported_url(url: str) -> bool:
    return bool(_HOST_PATTERN.match(url.strip()))


def _download_sync(url: str) -> tuple[Path, str]:
    """blocking yt-dlp 호출. (mp4_path, video_title) 반환."""
    if not _is_supported_url(url):
        raise YouTubeIngestError("유튜브 URL만 지원합니다 (youtube.com / youtu.be).")

    # caller가 파일을 읽으므로 with 블록(자동 close+delete) 패턴은 부적합.
    tmp = NamedTemporaryFile(delete=False, suffix=".mp4")  # noqa: SIM115
    tmp.close()
    out_path = Path(tmp.name)

    # 단일 ydl 컨텍스트로 메타→다운로드 분리: webpage/player JS 재fetch 방지.
    # noplaylist=True 없으면 watch?v=...&list=... URL이 재생목록 전체로 fan-out.
    opts: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(out_path),
        # 720p로 상한 — bare /best 폴백을 두지 않아 ≤5분이어도 4K 등 수 GB를 받는 걸 막는다.
        # ≤720p가 없으면 yt-dlp가 DownloadError를 던져 거부된다(고용량 다운로드보다 안전).
        "format": "best[ext=mp4][height<=720]/best[height<=720]",
        "overwrites": True,
        "noplaylist": True,
        "socket_timeout": 30,  # 응답 hang 가드
        "retries": 5,           # jitter 흡수, 영구 fail은 fast
        "fragment_retries": 5,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise YouTubeIngestError("영상 정보를 가져오지 못했습니다.")

            # 길이 미상(라이브 스트림 등)은 무한 다운로드 위험이라 거부.
            duration = int(info.get("duration") or 0)
            if duration <= 0:
                raise YouTubeIngestError(
                    "영상 길이를 확인할 수 없어 분석을 시작할 수 없습니다."
                )
            if duration > _MAX_DURATION_SEC:
                mins, secs = divmod(duration, 60)
                raise YouTubeIngestError(
                    f"영상이 너무 깁니다. {_MAX_DURATION_SEC // 60}분 이내만 지원합니다 "
                    f"(현재 {mins}분 {secs}초)."
                )
            if duration < _MIN_DURATION_SEC:
                raise YouTubeIngestError(
                    f"영상이 너무 짧습니다. 최소 {_MIN_DURATION_SEC}초 이상이어야 "
                    "분석할 수 있습니다."
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
