# Opal Studio - IMC / H&E OME-TIFF Image Viewer
__version__ = "0.1.5"


def _add_conda_library_path() -> None:
    """Make a conda env's CUDA libraries findable without `conda activate`.

    TensorFlow (StarDist, Mesmer, AI positivity) loads cudart / cublas / cudnn
    from the directories on PATH. `conda activate` adds the env's Library\\bin,
    where the cudatoolkit and cudnn packages put those DLLs, but launching the
    env's python or opal-studio.exe directly (the desktop shortcut, an IDE) does
    not, and TensorFlow then silently runs on the CPU. This runs on package
    import, so spawned worker processes get it too.
    """
    import os
    import sys

    if os.name != "nt":
        return
    prefix = sys.prefix
    if not os.path.isdir(os.path.join(prefix, "conda-meta")):
        return  # not a conda env
    # The same directories, in the same order, that `conda activate` prepends.
    wanted = [os.path.join(prefix, *parts) for parts in (
        (), ("Library", "mingw-w64", "bin"), ("Library", "usr", "bin"),
        ("Library", "bin"), ("Scripts",), ("bin",))]
    current = os.environ.get("PATH", "").split(os.pathsep)
    have = {os.path.normcase(os.path.normpath(p)) for p in current if p}
    missing = [d for d in wanted
               if os.path.isdir(d) and os.path.normcase(os.path.normpath(d)) not in have]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + current)


_add_conda_library_path()
