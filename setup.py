from setuptools import setup, find_packages

setup(
    name="opal-studio",
    version="0.1.0",
    description="Cross-platform spatial omics analysis toolkit",
    author="Tristan Whitmarsh",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "PySide6>=6.5.0",
        "tifffile",
        "imagecodecs",
        "numpy<2.0.0",
        "scikit-image",
        "pandas",
        "tensorflow==2.8.0",
        "torch>=2.0.0",
        "deepcell",
        "stardist",
        "cellpose",
        "omnipose",
        "instanseg-torch",
        "pywin32; sys_platform == 'win32'", # Required for creating windows shortcuts
    ],
    entry_points={
        "console_scripts": [
            "opal-studio=opal_studio.__main__:main",
        ],
    },
)
