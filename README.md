# Opal Studio

<img src="screenshot.jpg" width="100%">


**Opal Studio** is a high-performance, cross-platform image viewer and analysis application designed for highly multiplexed imaging data, including IMC (Imaging Mass Cytometry) and large OME-TIFF files. 

Built using PySide6 and leveraging a state-of-the-art Python data stack, Opal Studio offers native, responsive interfaces and robust multi-channel visualization, segmentation, and phenotyping workflows.

## Key Features

- **High-Performance Multi-Channel Viewing**: Load and dynamically manipulate dozens of high-resolution channels on the fly with native GPU-accelerated rendering.
- **Advanced Pre-processing**: Built-in normalization, CLAHE (Contrast Limited Adaptive Histogram Equalization), and morphological filters (Median, Tophat).
- **Multi-Engine AI Segmentation**: Deep integration with modern AI segmentation models:
    - **InstanSeg**: Fast, state-of-the-art nucleus and cell segmentation.
    - **StarDist (2D)**: Robust nuclear segmentation using star-convex polygons.
    - **Cellpose**: Integrated support for Cyto, Nuclei, and custom models.
    - **Omnipose**: Dedicated support for bacterial, plant, and high-res worm segmentation.
    - **Watershed**: Traditional marker-controlled region expansion for classical workflows.
- **Intelligent Mask Processing**: 
    - **Cell Sampler (Ubermasking)**: Geometrically merge results from multiple segmentation engines using intelligent strategies (Jaccard agreement, Population density, Area variance).
    - **Size Filtering**: Dynamically remove segmented regions based on area constraints.
    - **Watershed Expansion**: Expand nucleus masks into cell boundaries while maintaining neighborhood topology.
- **Phenotyping & Analysis**: 
    - **Interactive Phenotyping Grid**: Define cell types by mapping marker positivity (+/-) in an intuitive table.
    - **AI Cell Positivity**: Automated detection of marker expression using integrated deep learning models (Marker-CNN).
    - **Data Interoperability**: Full support for OME-TIFF mask export and CSV-based phenotyping configurations.
- **Commercial-Grade UI**: Fully responsive, dark-mode native interface built for professional research environments.

## Installation

Opal Studio relies on standard data science and imaging libraries. For a smooth setup, we recommend installing it within a **Conda environment**.

### 1. GPU & Driver Optimization (Critical for Cluster/Server Users)
Opal Studio dynamically uses your GPU for deep learning segmentation (Cellpose, InstanSeg, StarDist). 

To ensure PyTorch connects successfully with your graphics card:
* **Standard Modern PC/Workstation**: You can proceed directly to step 2 (pip will install the latest CUDA version automatically).
* **Older GPUs or University Clusters (e.g. CUDA 11.8 / A100 with older driver)**: 
  You should install the compatible PyTorch build *before* installing Opal Studio. For example, to target **CUDA 11.8**:
  ```bash
  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
  ```

---

### 2. Standard Installation (From Wheel or PyPI)
If you have downloaded/built a `.whl` distribution file (e.g. `opal_studio-0.1.0-py3-none-any.whl`), navigate to the directory and run:

```bash
# General installation (will download dependencies from PyPI)
pip install opal_studio-0.1.0-py3-none-any.whl

# If installing on a university cluster (enforcing CUDA 11.8 dependencies)
pip install opal_studio-0.1.0-py3-none-any.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

---

### 3. Developer / Source Installation
If you are developing Opal Studio or running it directly from source:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TristanWhitmarsh/opal-studio.git
   cd opal-studio
   ```

2. **Create and activate the Conda environment**:
   ```bash
   conda create -n opal-all python=3.9
   conda activate opal-all
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The `requirements.txt` is pre-configured to automatically download CUDA 11.8 compatible wheels for university servers).*

4. **Install Opal Studio in editable development mode**:
   ```bash
   pip install --no-deps -e .
   ```

## Usage

To launch Opal Studio, activate your environment and execute the application module:

```bash
conda activate opal-env
python -m opal_studio
```

### Creating a Desktop Launcher (Linux & Windows)

For convenience, Opal Studio can automatically generate a desktop launch button so you do not have to manually activate your environment and type the run command every time.

To create the desktop launcher, activate your virtual environment and run:

```bash
python -m opal_studio --create-launcher
```

#### Platform Specifics & Setup:

##### Linux (Ubuntu, Debian, Fedora, GNOME/KDE)
* **What it does**: Creates an `OpalStudio.desktop` launcher directly on your `~/Desktop`.
* **Terminal Logs Visibility**: The launcher is pre-configured with `Terminal=true` so that a terminal window automatically opens alongside the GUI. This allows you to monitor background image processing logs, AI model loading progress, and any warning or error outputs in real-time.
* **Making it Runnable (Crucial Step)**: Modern Linux desktop environments require desktop files on the desktop to be explicitly trusted before running.
  1. Navigate to your Desktop.
  2. **Right-click** on the newly created **Opal Studio** icon.
  3. Select **"Allow Launching"** (or mark it as trusted/executable in your environment).
  4. The standard shortcut icon will transform into the application icon, and you can now double-click it to run.

##### Windows
* **What it does**: It automatically creates a native Windows Desktop Shortcut (`Opal Studio.lnk`) using the Windows Script Host shell.
* It maps the shortcut to either the installed `opal-studio` command-line utility or the exact python interpreter of your currently activated virtual environment (running `python -m opal_studio`).

##### macOS
* Desktop launcher creation is not natively supported on macOS yet, but you can launch the app normally from the terminal.





## License

Opal Studio is licensed under the **MIT License** with the **Commons Clause**.

- ✅ **Free for use:** Researchers, academic labs, and companies are welcome to use Opal Studio for free to carry out their internal work.
- ✅ **Free to modify:** You may inspect, modify, and develop upon the code.
- ❌ **Commercial Restriction:** You may **not** sell the software, nor offer it as a paid hosted service or commercial product. 

See the `LICENSE` file for the exact details.

