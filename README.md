# Opal Studio

<img src="screenshot.jpg" width="100%">


**Opal Studio** is a high-performance, cross-platform image viewer and analysis application designed for highly multiplexed imaging data, including IMC (Imaging Mass Cytometry) and large OME-TIFF files. 

Built using PySide6 and leveraging a state-of-the-art Python data stack, Opal Studio offers native, responsive interfaces and robust multi-channel visualization, segmentation, and phenotyping workflows.

## Key Features

- **High-Performance Multi-Channel Viewing**: Load and dynamically manipulate dozens of high-resolution channels on the fly with native GPU-accelerated rendering.
- **Advanced Pre-processing**: Built-in normalization, CLAHE (Contrast Limited Adaptive Histogram Equalization), and morphological filters (Median, Tophat).
- **Multi-Engine AI Segmentation**: Deep integration with modern AI segmentation models:
    - **Watershed**: Traditional marker-controlled region expansion for classical workflows.
    - **InstanSeg**: Fast, state-of-the-art nucleus and cell segmentation.
    - **Mesmer (DeepCell)**: Integrated nuclear and whole-cell segmentation for multiplexed imaging.
    - **StarDist (2D)**: Robust nuclear segmentation using star-convex polygons.
    - **Cellpose**: Integrated support for Cyto, Nuclei, and custom models.
    - **Omnipose**: Dedicated support for bacterial, plant, and high-res worm segmentation.
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
```

#### Jupiter 3 — Virtual Environment (CUDA 11.8)
If you are installing on **Jupiter 3** (or any cluster node with a CUDA 11.8-compatible driver using a pre-existing virtual environment), execute the following commands in sequence:

```bash
source /storage/scratch.space/envs/opal-env/bin/activate
pip install --force-reinstall opal_studio-0.1.0-py3-none-any.whl --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
pip install "numpy==1.26.4" "requests>=2.31"
python -m opal_studio --create-launcher
```

#### Jupiter 4 — Conda Environment
If you are installing on **Jupiter 4**, activate the dedicated conda environment and install the wheel directly:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/scratch.space/envs/opal-env-j4
PIP_REQUIRE_VIRTUALENV=0 pip install opal_studio-0.1.0-py3-none-any.whl
```

> **Mesmer / TensorFlow GPU on Jupiter 4**  
> TensorFlow looks for `libcudart.so.11.0`, `libcublas.so.11`, `libcudnn.so.8`, etc. in `LD_LIBRARY_PATH`, but the default environment only adds the `compat/` and `CUPTI/` CUDA sub-directories — not the main `/usr/local/cuda/lib64/` directory where those libraries actually live. Without this path TensorFlow silently falls back to CPU, producing visibly degraded Mesmer results.  
> Add the following line **before** launching Opal Studio (or add it to your `~/.bashrc`):
> ```bash
> export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
> ```

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

#### Jupiter 4
On Jupiter 4, expose the main CUDA libraries before launching so that TensorFlow (used by Mesmer) can use the GPU:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/scratch.space/envs/opal-env-j4
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
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

## Performing Segmentation & Model Selection

Opal Studio features a comprehensive, multi-engine segmentation panel on the right side under the **Operations Panel**. 

### 1. How to Perform Segmentation

#### Channel and Model Selection
* **Select the Model**: Choose the segmentation tab corresponding to the engine you wish to use (e.g., *Watershed*, *InstanSeg*, etc.), then select the model from the **Model** dropdown.
* **Select the Channel**: Choose the input channel (e.g., DNA, DAPI, or another nuclear marker) from the **Channel** dropdown.

#### Region and Target Options
* **Full image vs. Visible region vs. Selected region**:
  * **Full image**: Runs the chosen segmentation engine over the entire image.
  * **Visible region**: Perfect for rapid testing. It segments only the portion of the image currently visible on your canvas.
  * **Selected region**: Runs segmentation inside the bounding box of the currently selected region polygon (drawn in the Regions tab). Only cells whose centroid falls inside the polygon are added to the mask.
* **New mask vs. Overwrite selected mask**:
  * **New mask**: Generates a brand-new mask channel in the left panel.
  * **Overwrite selected mask**: Overwrites the active, highlighted mask channel, allowing you to rapidly iterate and adjust parameters without cluttering your channels. When using **Selected region**, any existing cells inside the region are removed first, then the new detections are merged in.
* **Smart Border Handling**: When segmenting sub-regions, the backend automatically handles border cell segmentations. Cells that are cut off or touch boundaries are smartly handled and clipped, preventing edge artifacts and ensuring clean masks when tiling or zooming.

---

### 2. Model Selection Guide (with IMC Guidelines)

For **IMC (Imaging Mass Cytometry)** datasets, try using a **custom-trained IMC model** first (such as custom InstanSeg or Mesmer weights trained on IMC mass channels) to obtain modality-specific results.

#### Speed Comparison Reference
To help you select the right model for your workflow, here is a speed comparison benchmark (tested on a standard high-resolution dataset):

| Segmentation Engine | Execution Speed |
| :--- | :---: |
| **Watershed** | **1 sec** |
| **InstanSeg** | **10 sec** |
| **Mesmer (DeepCell)** | **26 sec** |
| **StarDist** | **60 sec** |
| **Cellpose** | **14 sec** |
| **Omnipose** | **25 sec** |

#### Rough Accuracy Ranking on IMC Data
When segmenting multiplexed IMC images, here is a rough objective accuracy assessment (from highest accuracy to lowest):

$$\text{StarDist} > \text{InstanSeg} > \text{Mesmer} > \text{Cellpose} > \text{Watershed} > \text{Omnipose}$$

Use this ranking along with the speed comparison to choose the optimal engine for your specific tissue and throughput requirements.

---
## License

Opal Studio is licensed under the **MIT License** with the **Commons Clause**.

- ✅ **Free for use:** Researchers, academic labs, and companies are welcome to use Opal Studio for free to carry out their internal work.
- ✅ **Free to modify:** You may inspect, modify, and develop upon the code.
- ❌ **Commercial Restriction:** You may **not** sell the software, nor offer it as a paid hosted service or commercial product. 

See the `LICENSE` file for the exact details.

