"""
Tests for model_store: downloading models on first use, without the network.

The download itself is replaced by a stand-in that writes whatever a real
connection might deliver — a complete archive, a truncated one, a corrupted one —
so the install logic is exercised exactly as it would be against GitHub.

Run with pytest, or directly::

    python tests/test_model_store.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opal_studio import model_store as ms  # noqa: E402

# Incompressible, like real weights. Highly repetitive test data compresses to a
# few hundred bytes of deflate stream in which some bit flips change nothing, so
# a corruption test built on it would not exercise the CRC check at all.
PAYLOAD = os.urandom(300_000)
KEY = "test/model"


def _archive(payload: bytes = PAYLOAD) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("weights.bin", payload)
    return buf.getvalue()


class _Sandbox:
    """An empty models directory, no source checkout, and a fake registry entry."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._env = os.environ.get(ms.ENV_VAR)
        os.environ[ms.ENV_VAR] = self.tmp.name
        self._pkg, self._fetch, self._waits = ms.PACKAGE_DIR, ms._fetch, ms.RETRY_WAITS
        ms.PACKAGE_DIR = Path(self.tmp.name) / "no_checkout"
        ms.RETRY_WAITS = (0, 0, 0)                   # retry without actually waiting
        ms.MODELS[KEY] = ms.ModelSpec(KEY, "http://example.invalid/m.zip",
                                      "test", "model", "test", "MIT")
        return self

    def leftovers(self) -> list[str]:
        root = ms.models_dir() / "test"
        return sorted(p.name for p in root.iterdir()) if root.exists() else []

    def __exit__(self, *exc):
        ms.MODELS.pop(KEY, None)
        ms.PACKAGE_DIR, ms._fetch, ms.RETRY_WAITS = self._pkg, self._fetch, self._waits
        if self._env is None:
            os.environ.pop(ms.ENV_VAR, None)
        else:
            os.environ[ms.ENV_VAR] = self._env
        self.tmp.cleanup()


def test_complete_download_is_installed():
    with _Sandbox() as box:
        ms._fetch = lambda url, dest, progress: Path(dest).write_bytes(_archive())
        path = ms.ensure(KEY)
        assert (path / "weights.bin").read_bytes() == PAYLOAD
        assert box.leftovers() == ["model"]          # no temp files


def test_installed_model_is_not_downloaded_again():
    with _Sandbox():
        calls = []

        def fetch(url, dest, progress):
            calls.append(url)
            Path(dest).write_bytes(_archive())

        ms._fetch = fetch
        ms.ensure(KEY)
        ms.ensure(KEY)
        assert len(calls) == 1


def test_truncated_download_is_rejected_and_left_nothing():
    with _Sandbox() as box:
        calls = []

        def fetch(url, dest, progress):
            calls.append(1)
            Path(dest).write_bytes(_archive()[:5000])
            raise ms.DownloadError("download stopped after 5,000 bytes")

        ms._fetch = fetch
        try:
            ms.ensure(KEY)
        except ms.DownloadError:
            pass
        else:
            raise AssertionError("a truncated download was accepted")
        assert len(calls) == len(ms.RETRY_WAITS) + 1  # every attempt used
        assert ms.find(KEY) is None
        assert box.leftovers() == []


def test_corrupt_archive_is_rejected():
    with _Sandbox() as box:
        def fetch(url, dest, progress):
            data = bytearray(_archive())
            data[len(data) // 2] ^= 0x01             # one bit, inside the data
            Path(dest).write_bytes(bytes(data))

        ms._fetch = fetch
        try:
            ms.ensure(KEY)
        except ms.DownloadError:
            pass
        else:
            raise AssertionError("a corrupt archive was installed")
        assert ms.find(KEY) is None
        assert box.leftovers() == []


def test_a_failed_attempt_is_retried():
    with _Sandbox():
        attempts = {"n": 0}

        def fetch(url, dest, progress):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("connection reset")
            Path(dest).write_bytes(_archive())

        ms._fetch = fetch
        assert (ms.ensure(KEY) / "weights.bin").exists()


def _http_error(code: int):
    import urllib.error
    return urllib.error.HTTPError("http://example.invalid/m.zip", code,
                                  "error", {}, io.BytesIO(b""))


def test_server_errors_are_retried_until_they_clear():
    # GitHub's release downloads fail intermittently with an immediate 504.
    with _Sandbox():
        attempts = {"n": 0}

        def fetch(url, dest, progress):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise _http_error(504)
            Path(dest).write_bytes(_archive())

        ms._fetch = fetch
        assert (ms.ensure(KEY) / "weights.bin").exists()
        assert attempts["n"] == 3


def test_not_found_fails_at_once():
    with _Sandbox():
        attempts = {"n": 0}

        def fetch(url, dest, progress):
            attempts["n"] += 1
            raise _http_error(404)

        ms._fetch = fetch
        try:
            ms.ensure(KEY)
        except ms.DownloadError as exc:
            assert "404" in str(exc)
        else:
            raise AssertionError("expected an error")
        assert attempts["n"] == 1                    # no pointless retries


def test_archive_wrapped_in_a_folder_is_unwrapped():
    with _Sandbox():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("model/weights.bin", PAYLOAD)
        ms._fetch = lambda url, dest, progress: Path(dest).write_bytes(buf.getvalue())
        assert (ms.ensure(KEY) / "weights.bin").exists()


def test_models_live_next_to_the_code_by_default():
    saved = os.environ.pop(ms.ENV_VAR, None)
    try:
        assert ms.models_dir() == Path(ms.__file__).resolve().parent / "models"
    finally:
        if saved is not None:
            os.environ[ms.ENV_VAR] = saved


def test_library_models_pass_through():
    assert ms.resolve("2D_versatile_fluo") == "2D_versatile_fluo"
    assert ms.resolve(None) is None


def test_unknown_missing_model_names_where_it_looked():
    with _Sandbox():
        try:
            ms.ensure("stardist/NoSuchModel")
        except ms.DownloadError as exc:
            assert "Looked in" in str(exc)
        else:
            raise AssertionError("expected an error")


def test_own_models_point_at_this_repository_release():
    own = [m for m in ms.MODELS.values() if m.source == "Opal Studio"]
    assert own, "no Opal Studio models registered"
    for m in own:
        assert m.url.startswith(ms.RELEASE_URL + "/"), m.url
        assert m.url.endswith(".zip"), m.url
        assert m.licence == "MIT"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"OK    {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
