# SAMPro3D-trial

> **3D point cloud processing is often computationally expensive.**  
> This repository contains my customized implementation of **SAMPro3D** for efficient and flexible 3D instance segmentation experiments.


https://github.com/user-attachments/assets/1b519fda-18f4-4163-9f64-f5433003dbbc


---

## 📄 Original Paper  
**SAMPro3D: Locating SAM Prompts in 3D for Zero-Shot Instance Segmentation**  
🔗 [Mutian Xu et al. – Official Project Page](https://mutianxu.github.io/sampro3d/)

---

## 📂 Repository Structure
```

SAMPro3D-trial/
├── room.ply                  # Example 3D point cloud input
├── room.xyz                  # Scene point cloud for matching
├── rgb_images/               # Captured RGB images (auto-created by the script)
├── depth_images/             # Captured depth images (auto-created by the script)
├── params/                   # Saved camera parameters (auto-created by the script)
├── saved_images/             # 2D projection plots (auto-created by the script)
├── sam2_hiera_l.yaml         # Configuration files for SAM2
├── sam2_hiera_large.pt       # Pre-trained SAM2 weights
├── main.py     # Main script containing all processing steps and functions
├── requirements.txt          # Dependencies list
└── README.md                 # Project documentation

````

---

## 🚀 Usage Instructions

### 1️⃣ **Installation**
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/SAMPro3D-trial.git
cd SAMPro3D-trial
pip install -r requirements.txt
````

---

### 2️⃣ **Prepare Your Data**

* Place your **input point cloud** (e.g., `room.ply`) and **scene cloud** (`room.xyz`) inside the `data/` folder.
* Ensure you have the SAM2 **checkpoint** (`sam2_hiera_large.pt`) and **config** (`sam2_hiera_l.yaml`) in the appropriate folders (`checkpoints/` and `configs/`).

---

### 3️⃣ **Run the Pipeline**

Execute the main script:

```bash
python src/sampro3d_trial.py
```

---

### 4️⃣ **Interactive Steps**

1. **Pick Points**

   * The Open3D visualizer opens.
   * Select **foreground points** (`shift + left-click`) inside the desired area, then press **Q**.
   * Repeat for **background points**.

2. **Capture Images**

   * The script will rotate the scene and capture **RGB** and **depth images** automatically.
   * Camera parameters will be saved as JSON in the `params/` folder.

3. **Segmentation & Mask Filtering**

   * SAM2 is applied to segment the object based on selected points.
   * Masks are filtered and overlaid on the RGB images for verification.

4. **Point Reprojection and Matching**

   * 2D points are reprojected back to 3D.
   * Matched points between the instance and scene clouds are filtered using `cKDTree`.

5. **Mesh Generation**

   * High-accuracy meshes are generated using Poisson reconstruction.
   * Smaller disconnected parts are removed, keeping only the largest connected component.

6. **Combine Results**

   * All matched point clouds are merged and saved as `combined_matched_original_pcd.ply`.

---

### 5️⃣ **Output**

* ✅ **Matched Point Clouds**: `matched_original_pcd_*.ply`
* ✅ **Combined Point Cloud**: `combined_matched_original_pcd.ply`
* ✅ **Filtered Masks & Overlays**: Saved in `saved_images/`
* ✅ **Captured Data**: RGB, depth, and parameters saved automatically

---

### 🧰 **Key Libraries Used**

* [Open3D](http://www.open3d.org/) – Point cloud visualization and processing
* [PyTorch](https://pytorch.org/) – Tensor computations and GPU acceleration
* [SciPy](https://scipy.org/) (`cKDTree`) – Fast nearest neighbor search
* [OpenCV](https://opencv.org/) – Image and contour operations
* [Matplotlib](https://matplotlib.org/) – Visualization
* [Pandas](https://pandas.pydata.org/) – Correlation and data analysis
* [SAM2](https://github.com/mutianxu/SAMPro3D) – Image segmentation backbone
