# Opal Studio

![Opal Studio Screenshot](screenshot.jpg)

**Opal Studio** is a fast, cross-platform image viewer and analysis application designed specifically for highly multiplexed imaging data, including IMC (Imaging Mass Cytometry) and large H&E OME-TIFF files. 

Built using PySide6 and leveraging a state-of-the-art Python data stack, Opal Studio offers native, responsive UIs and robust multi-channel visualization and segmentation workflows.

## Key Features

- **High-Performance Multi-Channel Viewing**: Load and dynamically manipulate dozens of high-resolution channels on the fly.
- **Advanced Pre-processing**: Built-in normalization and contrast limited adaptive histogram equalization (CLAHE).
- **AI Cell Segmentation**: Deep integration with modern AI segmentation models:
    - **StarDist (2D)** support for robust nuclear segmentation
    - **Cellpose** integration
    - **Watershed** marker-controlled region expansion
- **Phenotyping & Positivity Detection**: AI-based single-cell tagging and mask expansion capabilities natively built into the UI.
- **Fast, Native Interface**: Fully responsive, dark-mode styling built on native PySide6 UI components.

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
   conda activate opal-env
   ```

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

Use `File > Open Image` to load `.ome.tiff`, `.tiff`, and other standard formats. Utilize the left panel to configure visibility, brightness, and colors, and the right panel to orchestrate operations like segmentation and phenotyping.

## License

Opal Studio is licensed under the **MIT License** with the **Commons Clause**.

- ✅ **Free for use:** Researchers, academic labs, and companies are welcome to use Opal Studio for free to carry out their internal work.
- ✅ **Free to modify:** You may inspect, modify, and develop upon the code.
- ❌ **Commercial Restriction:** You may **not** sell the software, nor offer it as a paid hosted service or commercial product. 

See the `LICENSE` file for the exact details.
