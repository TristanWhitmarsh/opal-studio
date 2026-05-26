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

## Installation & Launch Guide

### 1. Default Installation (From Source)
For external users or standard installations, you can install Opal Studio by cloning the repository from GitHub and setting up a conda environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TristanWhitmarsh/opal-studio.git
   cd opal-studio
   ```

2. **Create and activate the Conda environment**:
   ```bash
   conda create -n opal-env python=3.9
   conda activate opal-env
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Opal Studio in editable development mode**:
   ```bash
   pip install --no-deps -e .
   ```

5. **Run the application**:
   ```bash
   python -m opal_studio
   ```

---

### 2. University Server Installation (Darkroom Setup)
If you are working on **Darkroom** (our university JupyterLab setup), you have to start the remote Linux desktop, open a terminal, and run:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/scratch.space/envs/opal-env-j4
python -m opal_studio --create-launcher
```

Then, simply double-click the **Opal Studio** icon on your desktop to launch!


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

