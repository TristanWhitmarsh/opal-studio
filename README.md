# Opal Studio

<img src="screenshot.jpg" width="100%">


**Opal Studio** is a high-performance, cross-platform image viewer and analysis application designed for highly multiplexed imaging data, including IMC (Imaging Mass Cytometry) and large OME-TIFF files. 

Built using PySide6 and leveraging a state-of-the-art Python data stack, Opal Studio offers native, responsive interfaces and robust multi-channel visualization, segmentation, and phenotyping workflows.

## Key Features

- **High-Performance Multi-Channel Viewing**: Load and dynamically manipulate dozens of high-resolution channels on the fly with native GPU-accelerated rendering.
- **Advanced Pre-processing**: Built-in normalization, CLAHE (Contrast Limited Adaptive Histogram Equalization), and morphological filters (Median, Tophat).
- **Multi-Engine AI Segmentation**: Deep integration with modern AI segmentation models:
    - **Mesmer (DeepCell)**: Advanced nuclear AND cytoplasmic segmentation for tissue data.
    - **InstanSeg**: Ultra-fast, state-of-the-art nucleus and cell segmentation.
    - **StarDist (2D)**: Robust nuclear segmentation using star-convex polygons.
    - **Cellpose**: Integrated support for Cyto, Nuclei, and custom models.
    - **Watershed**: Traditional marker-controlled region expansion for classical workflows.
- **Regional Segmentation Controls**:
    - **Visible Region Only**: Run heavy AI inference strictly on your current viewport to save time and memory.
    - **Full Image Mode**: Standard full-slide processing.
- **Intelligent Mask Management**: 
    - **New vs. Overwrite**: Generate new mask layers or surgically overwrite specific regions in existing masks.
    - **Population Sampling**: Geometrically merge results from multiple engines (Jaccard, population density, area variance).
- **Phenotyping & Analysis**: 
    - **Interactive Phenotyping Grid**: Define cell types by mapping marker positivity (+/-) in an intuitive table.
    - **AI Cell Positivity**: Automated detection of marker expression using integrated deep learning models (Marker-CNN).
- **Advanced Data Interoperability**: 
    - **OME-TIFF Export**: Export full hierarchical mask data.
    - **GeoJSON Export (QuPath)**: Export cell contours as standard `detection` GeoJSON features compatible with QuPath.
- **Under-the-Hood Optimization**: 
    - **Automatic Tiling & Batching**: Effortlessly handle 10k x 10k images via parallel tile-based inference.
    - **OpenCV-Accelerated Vectors**: Instantaneous contour generation using optimized C++ engines.
    - **Subsampled Normalization**: Rapid model scaling for large-scale datasets.

## Installation

Opal Studio relies on standard data science and imaging libraries. We recommend installing it within a Conda environment.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TristanWhitmarsh/opal-studio.git
   cd opal-studio
   ```

2. **Create and activate the environment**:
   Using the provided `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate opal-all
   ```

### ⚡ GPU Acceleration Note
To enable full GPU acceleration for all segmentation models on Windows, ensure you have the following versions installed in your environment:
* **TensorFlow (Mesmer/StarDist)**: `cudatoolkit=11.2` and `cudnn=8.1`.
* **PyTorch (Cellpose/InstanSeg)**: `torch+cu121` binaries.

3. **Install the package (Editable mode)**:
   ```bash
   pip install -e .
   ```

## Usage

To launch Opal Studio, activate your environment and execute the application module:

```bash
conda activate opal-env
python -m opal_studio
```

### Quick Start
1. **Load Image**: `File > Open Image` to load `.ome.tiff`, `.tiff`, or other standard formats.
2. **Visualize**: Use the left **Channel Panel** to toggle visibility and adjust colors.
3. **Segment**: Select an engine in the **Operations Panel**. Choose **Visible region only** for rapid testing or **Full image** for final results.
4. **Phenotype**: Switch to the **Phenotyping** tab in the center area to define cell populations.
5. **Export**: Use `File > Save Contours (GeoJSON)...` to move your vector data to QuPath or `File > Save Masks` for OME-TIFFs.

## License

Opal Studio is licensed under the **MIT License** with the **Commons Clause**.

- ✅ **Free for use:** Researchers, academic labs, and companies are welcome to use Opal Studio for free to carry out their internal work.
- ✅ **Free to modify:** You may inspect, modify, and develop upon the code.
- ❌ **Commercial Restriction:** You may **not** sell the software, nor offer it as a paid hosted service or commercial product. 

See the `LICENSE` file for the exact details.

