# FSDA-DG: Single Domain Generalization for Medical Image Segmentation with Few Source Domain Annotations


This repository is the official PyTorch implementation of our paper: **"FSDA-DG：Single Domain Generalization Medical Image Segmentation with Few Source Domain Annotations"**.

## 1. Installation

This code requires PyTorch 1.10 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```


## 2. Data preparation

We conduct datasets preparation following [CSDG](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization)

<details>
  <summary>
    <b>1) Abdominal MRI</b>
  </summary>

0. Download [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/) and put the `/MR` folder under `./data/CHAOST2/` directory

1. Converting downloaded data (T2 SPIR) to `nii` files in 3D for the ease of reading.

run `./data/abdominal/CHAOST2/s1_dcm_img_to_nii.sh` to convert dicom images to nifti files.

run `./data/abdominal/CHAOST2/png_gth_to_nii.ipynp` to convert ground truth with `png` format to nifti.

2. Pre-processing downloaded images

run `./data/abdominal/CHAOST2/s2_image_normalize.ipynb`

run `./data/abdominal/CHAOST2/s3_resize_roi_reindex.ipynb`

The processed dataset is stored in `./data/abdominal/CHAOST2/processed/`

</details>

<details>
  <summary>
    <b>1) Abdominal CT</b>
  </summary>

0. Download [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and put the `/img` and `/label` folders under `./data/SABSCT/CT/` directory

1.Pre-processing downloaded images

run `./data/abdominal/SABS/s1_intensity_normalization.ipynb` to apply abdominal window.

run `./data/abdominal/SABS/s2_remove_excessive_boundary.ipynb` to remove excessive blank region. 

run `./data/abdominal/SABS/s3_resample_and_roi.ipynb` to do resampling and roi extraction.
</details>

The details for cardiac datasets will be given later.

The processed datasets can be found at [processed datasets](). Download and unzip the file where the folder structure should look this:
To maintain anonymity, the detailed link will be updated later.

```none
SSL-DG
├── ...
├── data
│   ├── abdominal
│   │   ├── CHAOST2
│   │   │   ├── processed
│   │   ├── SABSCT
│   │   │   ├── processed
│   ├── cardiac
│   │   ├── processed
│   │   │   ├── bSSFP
│   │   │   ├── LGE
├── ...
```

## 3. Inference Using Pretrained Model
Download the [pretrained model]() and unzip the file where the folder structure should look this:
To maintain anonymity, the detailed link will be updated later.
```none
SSL-DG
├── ...
├── logs
│   ├── 2023-xxxx-xx-xx-xx
│   │   ├── checkpoints
│   │   │   ├── latest.pth
│   │   ├── configs
│   │   │   ├── xx.yaml
│       │── visuals 
│       │── test_visual 
├── ...
```

<details>
  <summary>
    <b>1)  Example for Cross-sequence Cardiac Dataset</b>
  </summary>

For direction bSSFP -> LEG with 50% annotated samples (DICE 85.87), run the command 
```bash
python test.py -r logs/2023-07-31T10-47-53_seed22_efficientUnet_bSSFP_to_LEG_labelnum_0.5 
```
For direction LEG -> BSSFP with 20% annotated samples (DICE 83.15), run the command 
```bash
test.py -r logs/2023-08-01T19-14-19_seed22_efficientUnet_LEG_to_BSSFP_labelnum_0.2
```
</details>


Each log contains visual results of the test images.


And more Pretrained models will be given later.

## 4. Training the model
To reproduce the performance, you need one 3080 GPU

<details>
  <summary>
    <b>1) Cross-modality Abdominal Dataset</b>
  </summary>

For direction CT -> MRI, run the command 
```bash
python main.py --base configs/efficientUnet_SABSCT_to_CHAOS.yaml --seed 22 --labeled_bs 0.5  --labelnum 0.1/0.2/0.5
```

For direction MRI -> CT, run the command 
```bash
python main.py --base configs/efficientUnet_CHAOS_to_SABSCT.yaml --seed 22 --labeled_bs 0.5  --labelnum 0.1/0.2/0.5
```
</details>

<details>
  <summary>
    <b>2)  Cross-sequence Cardiac Dataset</b>
  </summary>

For direction bSSFP -> LEG, run the command 
```bash
python main.py --base configs/efficientUnet_bSSFP_to_LEG.yaml --seed 22 --labeled_bs 0.5  --labelnum 0.1/0.2/0.5
```

For direction LEG -> bSSFP, run the command 
```bash
python main.py --base configs/efficientUnet_LEG_to_bSSFP.yaml --seed 22 --labeled_bs 0.5  --labelnum 0.1/0.2/0.5
```
</details>


## 5. Other comments
Here, we open source the initial version of FSDA-DG with the segmentation model containing two parallel networks.  Later, we will further open source more versions of FSDA-DG to facilitate community collaboration and advancement.  Subsequent versions will include a 2D segmentation model with three parallel networks as well as 3D image segmentation applications.


## Acknowledgements

Our codes are built upon [CSDG](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization), [SLAug](https://github.com/Kaiseem/SLAug), and [MC-Net](https://github.com/ycwu1997/MC-Net), thanks for their contribution to the community and the development of researches!



