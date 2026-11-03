# Opal Studio - IMC / H&E OME-TIFF Image Viewer
__version__ = "0.1.6"


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


def preload_conda_cuda11() -> None:
    """Make the conda env's CUDA 11 libraries visible to TensorFlow (Linux).

    Conda never puts its lib/ on LD_LIBRARY_PATH, so the env's cudatoolkit and
    cudnn are invisible to TensorFlow, which dlopens them by soname. Loading
    them globally satisfies those lookups, and keeps working when the system
    CUDA is a different version (e.g. 12).

    Call this only in processes that run TensorFlow, before importing it. The
    env's cuDNN 8.1 shares the soname libcudnn.so.8 with the cuDNN 8.9 that
    PyTorch bundles, so loading it into a PyTorch process breaks its convolutions.
    """
    import ctypes
    import os
    import sys

    if not sys.platform.startswith("linux"):
        return
    lib = os.path.join(sys.prefix, "lib")
    # Dependencies first: cublas needs cublasLt, cudnn needs its sub-libraries.
    for soname in (
            "libcudart.so.11.0", "libcublasLt.so.11", "libcublas.so.11",
            "libcufft.so.10", "libcurand.so.10", "libcusolver.so.11",
            "libcusparse.so.11", "libcudnn_ops_infer.so.8",
            "libcudnn_ops_train.so.8", "libcudnn_cnn_infer.so.8",
            "libcudnn_cnn_train.so.8", "libcudnn_adv_infer.so.8",
            "libcudnn_adv_train.so.8", "libcudnn.so.8"):
        path = os.path.join(lib, soname)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
