"""RED tests for main.source_is_available — webcam index vs file path."""
from main import source_is_available


def test_numeric_index_is_treated_as_webcam():
    assert source_is_available("0") is True
    assert source_is_available("1") is True


def test_existing_file_is_available(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    assert source_is_available(str(f)) is True


def test_missing_path_is_not_available():
    assert source_is_available("definitely_not_here_42.mp4") is False
