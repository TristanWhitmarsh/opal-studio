"""Where Opal Studio's model weights live, and fetching them on first use.

Two kinds of model, handled differently:

* **Opal Studio's own models** — the cell positivity CNN and the IMC / General
  segmentation models trained for this project. They are published as assets on a
  GitHub release of this repository (MIT, like the code) and downloaded the first
  time they are needed. They are never shipped inside the Python package: they come
  to ~550 MB, far past what belongs in a wheel.

* **Models that come with a segmentation library** — StarDist's, Cellpose's and
  InstanSeg's pretrained models. The library fetches and caches these itself, from
  the library's own source; Opal Studio does not redistribute them. The one
  exception is InstanSeg's ``single_channel_nuclei``: it is published by InstanSeg
  but missing from the model index of the installed InstanSeg, so the library cannot
  fetch it by name. It is fetched here instead — still from InstanSeg's own release,
  never re-hosted.

Downloaded models go into ``models/`` inside the installed package, next to the
code, so nothing is scattered elsewhere on the machine. ``OPAL_STUDIO_MODELS``
overrides the location if it is set — for shared storage on a cluster, say.

pip only removes the files it installed itself, so downloaded models survive
``pip install --force-reinstall``, and ``pip uninstall`` leaves the ``models/``
folder behind where the code was. To fetch a model again, delete it from the models
directory; it is downloaded afresh the next time it is used.

A download is written to a temporary file and only moved into place once it has
arrived in full and every file in the archive has passed its CRC check, so an
interrupted download can never be mistaken for an installed model.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO = "TristanWhitmarsh/opal-studio"
RELEASE_TAG = "models-v1"
RELEASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

#: Prefix marking a dropdown entry as a model this module provides. The main
#: window resolves it to a local path — downloading if needed — before a run.
STORE_PREFIX = "store:"

ENV_VAR = "OPAL_STUDIO_MODELS"

Progress = Callable[[int, int], None]        # (bytes done, bytes total — 0 if unknown)


@dataclass(frozen=True)
class ModelSpec:
    key: str        # also the model's path under the models directory
    url: str
    method: str     # which segmentation tab offers it ("cellpos" for positivity)
    name: str       # label shown in the dropdown
    source: str
    licence: str


def _own(key: str, method: str, name: str) -> ModelSpec:
    asset = key.replace("/", "-") + ".zip"
    return ModelSpec(key, f"{RELEASE_URL}/{asset}", method, name, "Opal Studio", "MIT")


MODELS: dict[str, ModelSpec] = {m.key: m for m in [
    _own("cellpos",          "cellpos",   "Cell positivity"),
    _own("stardist/General", "stardist",  "General"),
    _own("stardist/IMC",     "stardist",  "IMC"),
    _own("cellpose/General", "cellpose",  "General"),
    _own("cellpose/IMC",     "cellpose",  "IMC"),
    _own("instanseg/IMC",    "instanseg", "IMC"),
    _own("mesmer/IMC",       "mesmer",    "IMC"),
    ModelSpec(
        "instanseg/single_channel_nuclei",
        "https://github.com/instanseg/instanseg/releases/download/"
        "instanseg_models_v0.1.2/single_channel_nuclei.zip",
        "instanseg", "single_channel_nuclei", "InstanSeg", "Apache-2.0"),
]}

#: The positivity CNN inside the ``cellpos`` model.
POSITIVITY_WEIGHTS = "marker_cnn_epoch_200.h5"

#: Pretrained models each library fetches and caches for itself. Listed so that
#: "Download all models" can fetch them ahead of time for offline use.
LIBRARY_MODELS = {
    "stardist":  ["2D_versatile_fluo", "2D_paper_dsb2018", "2D_versatile_he"],
    "instanseg": ["fluorescence_nuclei_and_cells", "brightfield_nuclei"],
    "cellpose":  ["nuclei", "cyto", "cyto2", "cyto3"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Locations
# ──────────────────────────────────────────────────────────────────────────────

#: models/ next to this module — inside site-packages for a pip install, or in the
#: source tree for a checkout. The weights are not part of the wheel; they are
#: downloaded into here on first use.
PACKAGE_DIR = Path(__file__).resolve().parent / "models"


def models_dir() -> Path:
    """Where downloaded models are kept: models/ inside the package, next to the code.

    Keeping them with the code means nothing is left behind anywhere else — removing
    the opal_studio folder removes its models too.
    """
    override = os.environ.get(ENV_VAR)
    return Path(override).expanduser() if override else PACKAGE_DIR


def search_dirs() -> list[Path]:
    dirs = [models_dir()]
    if PACKAGE_DIR.is_dir() and PACKAGE_DIR.resolve() != dirs[0].resolve():
        dirs.append(PACKAGE_DIR)
    return dirs


def find(key: str) -> Optional[Path]:
    """The installed copy of a model, or None if it is not on this machine."""
    for base in search_dirs():
        path = base / key
        if path.is_dir() and any(path.iterdir()):
            return path
        if path.is_file():
            return path
    return None


def local_models(method: str) -> list[str]:
    """Model folders present for *method* that are not in the registry.

    Lets someone drop a model they trained themselves into the models directory
    and have it offered alongside the published ones.
    """
    known = {m.key for m in MODELS.values()}
    seen: list[str] = []
    for base in search_dirs():
        root = base / method
        if not root.is_dir():
            continue
        for sub in sorted(root.iterdir()):
            key = f"{method}/{sub.name}"
            if sub.is_dir() and key not in known and key not in seen:
                seen.append(key)
    return seen


def for_method(method: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.method == method]


def ref(key: str) -> str:
    """The dropdown value for a model this module provides."""
    return STORE_PREFIX + key


def is_ref(value) -> bool:
    return isinstance(value, str) and value.startswith(STORE_PREFIX)


# ──────────────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────────────

class DownloadError(RuntimeError):
    pass


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _user_agent() -> str:
    try:
        import importlib.metadata as md
        return f"opal-studio/{md.version('opal-studio')}"
    except Exception:
        return "opal-studio"


def _fetch(url: str, dest: Path, progress: Optional[Progress]) -> None:
    """Stream *url* to *dest*, failing if fewer bytes arrive than were promised."""
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    if total and done != total:
        raise DownloadError(f"download stopped after {done:,} of {total:,} bytes")


def _install(archive: Path, target: Path) -> None:
    """Verify an archive and move its contents to *target* in one step."""
    staging = target.parent / f".{target.name}.installing"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()           # checks every member against its CRC
            if bad is not None:
                raise DownloadError(f"archive is corrupt at {bad}")
            zf.extractall(staging)

        # Accept either layout: files at the top of the archive, or wrapped in a
        # single folder.
        entries = list(staging.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
        if not any(root.iterdir()):
            raise DownloadError("archive is empty")

        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        os.replace(root, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


#: Seconds to wait before each retry. GitHub's release downloads fail
#: intermittently with an immediate 504 that clears within seconds — measured on
#: this project's own release, assets needed two or three tries a few seconds
#: apart — so a quick second attempt alone is not enough.
RETRY_WAITS = (3, 8, 15)


def _retryable(exc: Exception) -> bool:
    """Whether trying again could help.

    Server errors, rate limiting, dropped connections and damaged downloads are
    usually transient. Any other HTTP error — a 404 above all — will not fix
    itself, so it is reported at once rather than after a minute of retries.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return True


def download(key: str, progress: Optional[Progress] = None,
             attempts: Optional[int] = None) -> Path:
    """Fetch and install a registered model into the models directory."""
    spec = MODELS.get(key)
    if spec is None:
        raise DownloadError(f"no download is known for '{key}'")

    target = models_dir() / key
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloadError(
            f"cannot write models to {target.parent} ({exc}). Opal Studio keeps its "
            f"models next to its code, so it needs to be installed somewhere you can "
            f"write to — a conda env or virtualenv — or set {ENV_VAR} to a writable "
            f"directory.") from exc
    part = target.parent / f".{target.name}.part"

    if attempts is None:
        attempts = len(RETRY_WAITS) + 1

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _fetch(spec.url, part, progress)
            _install(part, target)
            return target
        except Exception as exc:        # server error, network drop, truncation, bad CRC
            last = exc
            if attempt == attempts or not _retryable(exc):
                break
            wait = RETRY_WAITS[min(attempt - 1, len(RETRY_WAITS) - 1)]
            print(f"[Models] {key}: attempt {attempt} of {attempts} failed ({exc}); "
                  f"retrying in {wait}s")
            time.sleep(wait)
        finally:
            part.unlink(missing_ok=True)

    if isinstance(last, urllib.error.HTTPError) and last.code == 404:
        raise DownloadError(f"{key} was not found at {spec.url} (HTTP 404)") from last
    raise DownloadError(f"could not download {key} from {spec.url} after "
                        f"{attempt} attempt(s): {last}") from last


def ensure(key: str, progress: Optional[Progress] = None) -> Path:
    """The local copy of a model, downloading it first if it is not here yet."""
    with _lock(key):
        found = find(key)
        if found is not None:
            return found
        if key not in MODELS:
            raise DownloadError(
                f"model '{key}' is not installed and there is no download for it. "
                f"Looked in: {', '.join(str(d) for d in search_dirs())}")
        print(f"[Models] {key} not found locally — downloading from {MODELS[key].url}")
        return download(key, progress)


def entrypoint(key: str, path: Path) -> Path:
    """What a segmentation method loads from an installed model folder.

    StarDist and InstanSeg load a folder; Cellpose and Mesmer load one file inside
    it; the positivity model is a single named file.
    """
    method = key.split("/", 1)[0]
    if key == "cellpos":
        return path / POSITIVITY_WEIGHTS if path.is_dir() else path
    if method == "mesmer" and path.is_dir():
        keras = sorted(path.glob("*.keras"))
        if not keras:
            raise DownloadError(f"no .keras file in {path}")
        return keras[0]
    if method == "cellpose" and path.is_dir():
        # A Cellpose model is one extension-less weights file; skip anything that
        # is plainly documentation or an image.
        skip = (".json", ".txt", ".md", ".ipynb", ".png", ".jpg", ".jpeg", ".npy")
        files = [p for p in path.iterdir() if p.is_file() and not p.name.endswith(skip)]
        if not files:
            raise DownloadError(f"no Cellpose weights in {path}")
        return max(files, key=lambda p: p.stat().st_size)
    return path


def resolve(value, progress: Optional[Progress] = None):
    """Turn a dropdown value into what a segmentation method loads.

    A ``store:`` reference becomes a local path, downloaded if needed; anything
    else is passed through untouched — a library model name, or None.
    """
    if not is_ref(value):
        return value
    key = value[len(STORE_PREFIX):]
    return str(entrypoint(key, ensure(key, progress)))


# ──────────────────────────────────────────────────────────────────────────────
# Download everything
# ──────────────────────────────────────────────────────────────────────────────

def _prefetch_worker(family: str, names: list[str], queue) -> None:
    """Have one library fetch its own pretrained models, in its own process.

    TensorFlow (StarDist) and PyTorch (Cellpose, InstanSeg) are kept in separate
    processes for the same reason segmentation is: their native libraries clash.
    """
    for name in names:
        try:
            if family == "stardist":
                from stardist.models import StarDist2D
                StarDist2D.from_pretrained(name)
            elif family == "instanseg":
                from opal_studio.segmentation_engine import ensure_instanseg_py39_compat
                ensure_instanseg_py39_compat()
                from instanseg.utils.utils import download_model
                download_model(name, verbose=False)
            elif family == "cellpose":
                from cellpose import models as cp
                cp.model_path(name)
                cp.size_model_path(name)
            queue.put((f"{family}/{name}", "ok", ""))
        except Exception as exc:
            queue.put((f"{family}/{name}", "failed", str(exc)))


def download_all(log: Callable[[str], None] = print,
                 progress: Optional[Callable[[str, int, int], None]] = None) -> dict:
    """Fetch every model Opal Studio can use, so it works without internet later.

    Returns {model: status}. Nothing already present is fetched again.
    """
    results: dict[str, str] = {}

    for key in MODELS:
        where = find(key)
        if where is not None:
            results[key] = f"already installed ({where})"
            log(f"{key}: already installed")
            continue
        log(f"{key}: downloading…")
        try:
            path = download(key, (lambda d, t, k=key: progress(k, d, t)) if progress else None)
            results[key] = f"downloaded ({path})"
            log(f"{key}: done")
        except Exception as exc:
            results[key] = f"FAILED: {exc}"
            log(f"{key}: FAILED — {exc}")

    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    for family, names in LIBRARY_MODELS.items():
        log(f"{family}: fetching {', '.join(names)} through {family}…")
        queue = ctx.Queue()
        proc = ctx.Process(target=_prefetch_worker, args=(family, names, queue))
        proc.start()
        got = 0
        while got < len(names):
            try:
                name, status, detail = queue.get(timeout=900)
            except Exception:
                break
            got += 1
            results[name] = status if status == "ok" else f"FAILED: {detail}"
            log(f"{name}: {'done' if status == 'ok' else 'FAILED — ' + detail}")
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
        for name in names:
            results.setdefault(f"{family}/{name}", "FAILED: no response from worker")

    return results
