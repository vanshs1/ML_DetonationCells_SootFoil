# A Machine Learning Based Approach for Statistical Analysis of Detonation Cells from Soot Foils

> **This work present a generalized algorithm that automatically detects, segments, and precisely measures detonation cells in soot foil images. Using advances in cellular biology segmentation models ([CellPose](https://cellpose.readthedocs.io/en/latest/index.html)), the algorithm is designed to accurately extract cellular patterns without a training procedure or dataset, which is a significant challenge in detonation research.**

Authors: 
    - Vansh Sharma, Michael Ullman and Venkat Raman

Affiliation: 
    - [APCL](https://sites.google.com/umich.edu/apcl/home?authuser=0) Group 
    - Department of Aerospace Engineering, University of Michigan, Ann Arbor

---

## 📜 Citation
If you use this work in your research, please cite it as follows:

```
@article{yourcitation2025,
  author    = {Your Name and Co-Authors},
  title     = {UPDATE},
  journal   = {UPDATE},
  year      = {2025},
  volume    = {X},
  number    = {Y},
  pages     = {1-10},
  doi       = {10.XXXX/yourdoi}
}
```

---

## 🚀 Installation

Clone the repository and install dependencies:
Python v3.9.x and above

```bash
# Create virtual environment (optional but recommended)
python -m venv MLsootfoil
source MLsootfoil/bin/activate 

# Install required packages
pip install -r package.list
```

---

## 🛠️ Usage

### 1️⃣ **Run Sootfoil Segmentation**

```bash
python main_sootFoilSeg.py
```

### 2️⃣ **Modify Configuration**

Edit `config.py` to customize:

- `ENABLE_PREPROCESSING`: Toggle pre-processing enhancements.
- `ENABLE_DENOISING`: Enable ML-based noise reduction.
- `NUM_LINES`: Adjust the number of analysis lines.
- `AXIS_SELECTION`: Set line analysis direction (`"vertical"` or `"horizontal"`).

Example:

```python
ENABLE_DENOISING = True
NUM_LINES = 4
AXIS_SELECTION = "horizontal"
```
Test data (case 10a) is available in the example folder.

1. First, run main_sootFoilSeg.py to generate statistics without applying pre-processing.

2. Next, update the output folder name, then modify config.py by setting:
ENABLE_PREPROCESSING = True
OVERLAY_COUNT = 8
3. Run main_sootFoilSeg.py again to process the images with pre-processing enabled.

By comparing both outputs, you can observe the impact of pre-processing.
Note: Pre-processing may not be required for all cases. Users can experiment with different models to achieve more precise results. For further insights, refer to Table 1 in the article.

---

## 📊 Visualizations

- **Cell Segmentation**: Overlays detected cell boundaries.
- **Histograms**: Dx, Dy, and cell area distributions along defined lines.
- **Pre/Post Processing Comparison**: See how denoising and enhancements improve segmentation.

> **Example Output:**

![Example Segmentation](https://raw.githubusercontent.com/yourusername/image-analysis/main/examples/example_from_Sharma_et_al_CnF.png)


---

## 🤝 Contributing

Pull requests are closed for now! For major changes, please email.

---

##  License

see the `LICENSE` file for details.

---



