"""_script_lib 순수 함수 단위 테스트 — model_tag · 경로 · 라벨 필터 · dump 가드."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from vidoctor.eval._script_lib import (
    existing_file,
    eval_dump_path,
    experiment_name,
    filter_labels_by_dim,
    metrics_to_dict,
    model_tag,
    transcript_cache_path,
    write_eval_dump,
)
from vidoctor.eval.labels import GoldenLabel
from vidoctor.eval.metrics import DimensionMetrics

# ---------------------------------------------------------------------------
# existing_file — argparse type validator
# ---------------------------------------------------------------------------


def test_existing_file_returns_path_for_real_file(tmp_path: Path):
    f = tmp_path / "video.mp4"
    f.write_text("")
    assert existing_file(str(f)) == f


def test_existing_file_raises_on_missing(tmp_path: Path):
    with pytest.raises(argparse.ArgumentTypeError, match="찾을 수 없습니다"):
        existing_file(str(tmp_path / "nope.mp4"))


def test_existing_file_raises_on_directory(tmp_path: Path):
    with pytest.raises(argparse.ArgumentTypeError, match="파일이 아닙니다"):
        existing_file(str(tmp_path))


# ---------------------------------------------------------------------------
# model_tag — env var → 캐시 키
# ---------------------------------------------------------------------------


def test_model_tag_default_when_env_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VIDOCTOR_WHISPER_MODEL", raising=False)
    assert model_tag() == "default"


def test_model_tag_default_when_env_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDOCTOR_WHISPER_MODEL", "")
    assert model_tag() == "default"


def test_model_tag_uses_basename_for_path(monkeypatch: pytest.MonkeyPatch):
    # HuggingFace 모델 ID나 로컬 경로의 마지막 segment만 채택.
    monkeypatch.setenv("VIDOCTOR_WHISPER_MODEL", "openai/whisper-large-v3")
    assert model_tag() == "whisper-large-v3"


def test_model_tag_plain_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDOCTOR_WHISPER_MODEL", "medium")
    assert model_tag() == "medium"


def test_model_tag_empty_basename_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
):
    # Path("/").name == "" → `or "default"` 가드가 살아있는지 확인.
    monkeypatch.setenv("VIDOCTOR_WHISPER_MODEL", "/")
    assert model_tag() == "default"


# ---------------------------------------------------------------------------
# transcript_cache_path / eval_dump_path / experiment_name — 경로·이름 합성
# ---------------------------------------------------------------------------


def test_transcript_cache_path_embeds_model_tag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDOCTOR_WHISPER_MODEL", "medium")
    p = transcript_cache_path(Path("/tmp/lecture.mp4"))
    assert p.name == "transcript_lecture_medium.json"
    assert p.parent.name == "inputs"


def test_transcript_cache_path_isolates_models(monkeypatch: pytest.MonkeyPatch):
    # 두 모델의 캐시가 같은 경로에 안 떨어지는지 — model_tag 시스템의 핵심 불변식.
    monkeypatch.setenv("VIDOCTOR_WHISPER_MODEL", "medium")
    medium_path = transcript_cache_path(Path("/tmp/lecture.mp4"))
    monkeypatch.setenv("VIDOCTOR_WHISPER_MODEL", "openai/whisper-large-v3")
    large_path = transcript_cache_path(Path("/tmp/lecture.mp4"))
    assert medium_path != large_path


def test_eval_dump_path_layout():
    p = eval_dump_path("filler", "lecture", "stage1_lecture")
    assert p.name == "lecture_stage1_lecture.json"
    assert p.parent.name == "filler"


def test_experiment_name_format():
    assert experiment_name("filler") == "vidoctor-filler"
    assert experiment_name("dead_zone") == "vidoctor-dead_zone"


# ---------------------------------------------------------------------------
# filter_labels_by_dim — 필터 + warning 사이드이펙트
# ---------------------------------------------------------------------------


def test_filter_labels_by_dim_returns_matching(caplog: pytest.LogCaptureFixture):
    labels = [
        GoldenLabel(start=1.0, end=2.0, dimension="filler"),
        GoldenLabel(start=10.0, end=15.0, dimension="cps"),
        GoldenLabel(start=20.0, end=22.0, dimension="filler"),
    ]
    with caplog.at_level("WARNING"):
        out = filter_labels_by_dim(labels, "filler")
    assert len(out) == 2
    assert all(lbl.dimension == "filler" for lbl in out)
    # 매칭이 있으면 warning은 발생하지 않아야 함.
    assert "라벨이 0개" not in caplog.text


def test_filter_labels_by_dim_empty_logs_warning(caplog: pytest.LogCaptureFixture):
    labels = [GoldenLabel(start=1.0, end=2.0, dimension="filler")]
    with caplog.at_level("WARNING"):
        out = filter_labels_by_dim(labels, "gaze")
    assert out == []
    assert "gaze 라벨이 0개" in caplog.text


def test_filter_labels_by_dim_empty_input_list(caplog: pytest.LogCaptureFixture):
    # 라벨 자체가 비어있는 케이스 (CSV가 비었거나 외부 호출 실수).
    with caplog.at_level("WARNING"):
        out = filter_labels_by_dim([], "filler")
    assert out == []
    assert "filler 라벨이 0개" in caplog.text


# ---------------------------------------------------------------------------
# metrics_to_dict — DimensionMetrics → MLflow log_metrics 호환 dict
# ---------------------------------------------------------------------------


def test_metrics_to_dict_includes_iou_by_default():
    m = DimensionMetrics(dimension="cps", tp=2, fp=1, fn=1, iou_sum=1.6)
    out = metrics_to_dict(m)
    assert out["tp"] == 2
    assert out["fp"] == 1
    assert out["fn"] == 1
    assert "precision" in out
    assert "recall" in out
    assert "f1" in out
    assert out["temporal_iou_mean"] == pytest.approx(0.8)


def test_metrics_to_dict_excludes_iou_when_flag_off():
    # filler는 IoU가 의미 없으니 include_iou=False로 호출.
    m = DimensionMetrics(dimension="filler", tp=1, fp=0, fn=0)
    out = metrics_to_dict(m, include_iou=False)
    assert "temporal_iou_mean" not in out


# ---------------------------------------------------------------------------
# write_eval_dump — 충돌 가드 + force 덮어쓰기
# ---------------------------------------------------------------------------


def test_write_eval_dump_creates_parent_and_writes(tmp_path: Path):
    out = tmp_path / "nested" / "result.json"
    write_eval_dump(out, {"k": "v"}, force=False)
    assert json.loads(out.read_text()) == {"k": "v"}


def test_write_eval_dump_raises_on_collision_without_force(tmp_path: Path):
    out = tmp_path / "result.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError, match="이미 존재"):
        write_eval_dump(out, {"k": "v"}, force=False)


def test_write_eval_dump_overwrites_with_force(tmp_path: Path):
    out = tmp_path / "result.json"
    out.write_text("{}")
    write_eval_dump(out, {"k": "v"}, force=True)
    assert json.loads(out.read_text()) == {"k": "v"}


def test_write_eval_dump_preserves_korean_unicode(tmp_path: Path):
    # ensure_ascii=False 플래그 회귀 가드. True로 바뀌면 한글이 \uXXXX escape로 깨짐.
    out = tmp_path / "result.json"
    data = {"note": "음·자·어 burst", "dimension": "filler"}
    write_eval_dump(out, data, force=False)
    raw = out.read_text()
    assert "음·자·어 burst" in raw
    assert "\\u" not in raw
    assert json.loads(raw) == data
