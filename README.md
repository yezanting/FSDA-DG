# FSDA-DG: Single Domain Generalization for Medical Image Segmentation with Few Source Domain Annotations


This repository is the official PyTorch implementation of our paper: **"FSDA-DG：Single Domain Generalization Medical Image Segmentation with Few Source Domain Annotations"**.

# Main framework overview and results

![Main framework](fig3_revised.png#gh-light-mode-only)

![Results](fig21.png#gh-light-mode-only)

![Main framework](fig1_white_background.png#gh-dark-mode-only)

![Results](fig2_white_background.png#gh-dark-mode-only)


## 📢 News
- **[2025.06.17] FSDA-DG has accepted by *Medical Image Analysis* (MedIA) ！🎉**
- **[2024.11.10]** We open-sourced a simply FSDA-DG code！🎉**


## 🔧 1. Installation

First, clone this repository and navigate to the project directory:
```bash
git clone https://github.com/yezanting/FSDA-DG.git
cd FSDA-DG
```

Next, install the required dependencies. We recommend using a virtual environment.
```bash
pip install -r requirements.txt
```

---

## 📦 2. Data Preparation

We follow the data preparation pipeline from [CSDG]. Please download the datasets and process them as described below.

<details>
  <summary><strong>Abdominal Datasets (CT & MRI)</strong></summary>

  #### Abdominal MRI
  1. Download the [Combined Healthy Abdominal Organ Segmentation (CHAOS) dataset].
  2. Place the downloaded `/MR` folder into the `./data/CHAOST2/` directory.
  3. Run the provided scripts to convert and preprocess the data:
     ```bash
     # Convert DICOM images to NIFTI format
     bash ./data/abdominal/CHAOST2/s1_dcm_img_to_nii.sh
     # Convert PNG ground truth masks to NIFTI format
     python ./data/abdominal/CHAOST2/png_gth_to_nii.ipynp
     # Normalize images and extract Region of Interest (ROI)
     python ./data/abdominal/CHAOST2/s2_image_normalize.ipynb
     python ./data/abdominal/CHAOST2/s3_resize_roi_reindex.ipynb
     ```
  The processed data will be saved in `./data/abdominal/CHAOST2/processed/`.

  #### Abdominal CT
  1. Download the [Synapse Multi-atlas Abdominal Segmentation dataset].
  2. Place the `/img` and `/label` folders into the `./data/SABSCT/CT/` directory.
  3. Run the preprocessing scripts:
     ```bash
     python ./data/abdominal/SABS/s1_intensity_normalization.ipynb
     python ./data/abdominal/SABS/s2_remove_excessive_boundary.ipynb
     python ./data/abdominal/SABS/s3_resample_and_roi.ipynb
     ```
  The processed data will be saved in `./data/abdominal/SABSCT/processed/`.

</details>

<details>
  <summary><strong>Cardiac Datasets (bSSFP & LGE)</strong></summary>
  
  (Detailed instructions for the cardiac datasets will be provided soon.)

</details>

> **Note**: For convenience, we will provide a download link for the fully processed datasets and our pretrained models at a later date.

The final data directory structure should look like this:
```
FSDA-DG/
├── data/
│   ├── abdominal/
│   │   ├── CHAOST2/
│   │   │   └── processed/
│   │   └── SABSCT/
│   │       └── processed/
│   └── cardiac/
│       └── processed/
│           ├── bSSFP/
│           └── LGE/
└── ...
```

---

## 🚀 3. Training

All training configurations are defined in the `.yaml` files within the `configs/` directory. You can start training with a single command. A GPU like the NVIDIA 3080 is recommended.

<details>
  <summary><strong>Cross-modality Abdominal Segmentation</strong></summary>
  
  - **Direction: CT -> MRI** (Train on Synapse, test on CHAOS)
    ```bash
    # Use --labelnum to specify the fraction of labeled data (e.g., 0.1 for 10%)
    python main.py --base configs/efficientUnet_SABSCT_to_CHAOS.yaml --seed 22 --labeled_bs 0.5 --labelnum 0.1
    ```

  - **Direction: MRI -> CT** (Train on CHAOS, test on Synapse)
    ```bash
    python main.py --base configs/efficientUnet_CHAOS_to_SABSCT.yaml --seed 22 --labeled_bs 0.5 --labelnum 0.1
    ```
</details>

<details>
  <summary><strong>Cross-sequence Cardiac Segmentation</strong></summary>

  - **Direction: bSSFP -> LGE**
    ```bash
    python main.py --base configs/efficientUnet_bSSFP_to_LEG.yaml --seed 22 --labeled_bs 0.5 --labelnum 0.2
    ```

  - **Direction: LGE -> bSSFP**
    ```bash
    python main.py --base configs/efficientUnet_LEG_to_bSSFP.yaml --seed 22 --labeled_bs 0.5 --labelnum 0.2
    ```
</details>

---

## 📊 4. Inference

Download our pretrained models and unzip them into the `logs/` directory.
> **Note**: The download link for pretrained models will be available soon.

Run the following commands to evaluate the models on the test set.

<details>
  <summary><strong>Example: Cross-sequence Cardiac Segmentation</strong></summary>
  
  - **Direction: bSSFP -> LGE** (with 50% labeled source samples, DICE 85.87)
    ```bash
    python test.py -r logs/2023-07-31T10-47-53_seed22_efficientUnet_bSSFP_to_LEG_labelnum_0.5
    ```

  - **Direction: LGE -> bSSFP** (with 20% labeled source samples, DICE 83.15)
    ```bash
    python test.py -r logs/2023-08-01T19-14-19_seed22_efficientUnet_LEG_to_BSSFP_labelnum_0.2
    ```
Visual segmentation results for each test case will be saved in the corresponding log directory.

</details>

---


## 🤝 Acknowledgements

Our codes are built upon [CSDG](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization), [SLAug](https://github.com/Kaiseem/SLAug), and [MC-Net](https://github.com/ycwu1997/MC-Net), thanks for their contribution to the community and the development of researches!



