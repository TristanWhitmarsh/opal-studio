"""Package Opal Studio's own models as GitHub release assets.

Run from the repository root with the models present under opal_studio/models/:

    python tools/build_model_release.py

Writes one zip per model to model_release/<tag>/, named exactly as
opal_studio/model_store.py expects to download them, plus RELEASE_NOTES.md to
paste in as the release description. Each zip carries the repository's LICENSE.

Then create a release with that tag on GitHub and upload every .zip unchanged.

Only files a model actually loads are packaged. The exclusions below were each
checked against the loader:
  * stardist/*/weights_last.h5 — csbdeep loads the weights file with "best" in
    its name when one exists, so this is never read.
  * *.ipynb — a notebook is not part of a model.
InstanSeg's experiment_log.csv is kept: InstanSeg rebuilds the network from it.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opal_studio import model_store as ms  # noqa: E402

SOURCE = ROOT / "opal_studio" / "models"
LICENSE = ROOT / "LICENSE"

EXCLUDE_NAMES = {"weights_last.h5"}
EXCLUDE_SUFFIXES = {".ipynb"}


def files_for(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return sorted(p for p in src.rglob("*")
                  if p.is_file()
                  and p.name not in EXCLUDE_NAMES
                  and p.suffix not in EXCLUDE_SUFFIXES
                  and "__pycache__" not in p.parts)


def build(tag: str = ms.RELEASE_TAG) -> Path:
    out = ROOT / "model_release" / tag
    out.mkdir(parents=True, exist_ok=True)
    if not LICENSE.exists():
        raise SystemExit(f"no LICENSE at {LICENSE}")

    rows = []
    for spec in ms.MODELS.values():
        if spec.source != "Opal Studio":
            continue                         # third-party: never re-hosted
        asset = spec.url.rsplit("/", 1)[1]
        src = SOURCE / spec.key
        if not src.exists():
            raise SystemExit(f"missing {src} — cannot package {spec.key}")

        files = files_for(src)
        if not files:
            raise SystemExit(f"nothing to package in {src}")

        dest = out / asset
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in files:
                zf.write(f, f.relative_to(src) if src.is_dir() else f.name)
            zf.write(LICENSE, "LICENSE")

        # Read it back the way the app will: every member must pass its CRC.
        with zipfile.ZipFile(dest) as zf:
            bad = zf.testzip()
            if bad:
                raise SystemExit(f"{asset}: {bad} failed its CRC check")
            names = zf.namelist()

        size = dest.stat().st_size
        rows.append((spec, asset, size, [n for n in names if n != "LICENSE"]))
        print(f"  {asset:24s} {size / 1e6:8.1f} MB   {', '.join(n for n in names if n != 'LICENSE')}")

    notes = out / "RELEASE_NOTES.md"
    lines = [
        f"# Opal Studio models — {tag}",
        "",
        "Model weights used by Opal Studio. The application downloads these "
        "automatically the first time each one is needed; they can also be fetched "
        "all at once from **File → Download All Models**.",
        "",
        "Downloaded models are kept in `opal_studio/models/` inside the installed "
        "package, next to the code.",
        "",
        "| Asset | Model | Used by | Size | Contents |",
        "|---|---|---|---|---|",
    ]
    for spec, asset, size, names in rows:
        lines.append(f"| `{asset}` | {spec.name} | {spec.method} | "
                     f"{size / 1e6:.1f} MB | {', '.join(f'`{n}`' for n in names)} |")
    lines += [
        "",
        "## Licence",
        "",
        "Released under the MIT licence, the same as the Opal Studio code. A copy "
        "of the licence is included in each archive.",
        "",
        "Pretrained models that ship with StarDist, Cellpose and InstanSeg are not "
        "part of this release; those libraries download them from their own sources "
        "under their own licences. See MODELS.md in the repository.",
    ]
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    where = build()
    total = sum(p.stat().st_size for p in where.glob("*.zip"))
    print(f"\nwrote {len(list(where.glob('*.zip')))} assets, {total / 1e6:.0f} MB, to {where}")
    print(f"release notes: {where / 'RELEASE_NOTES.md'}")
