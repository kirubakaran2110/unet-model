👇

🧠 U-Net Image Segmentation Project
📘 Overview

This project implements a U-Net-based image segmentation model using PyTorch.
The model takes an input image and predicts a segmentation mask, highlighting the regions of interest in the image.
U-Net is widely used for biomedical image segmentation, road extraction, and object localization tasks.

📂 Folder Structure
```
unet-segmentation/
│
├── data/
│   ├── images/       # Input images (e.g., im_0001.png)
│   ├── masks/        # Corresponding masks (e.g., m_0001.png)
│
├── checkpoints/
│   └── unet_final.pth   # Trained model weights
│
├── train_unet.ipynb     # Training notebook
├── inference_unet.ipynb # Inference / testing notebook
├── requirements.txt     # Dependencies list
└── README.md            # Project documentation
```
⚙️ Implementation Details

Framework: PyTorch
Architecture: U-Net (Encoder–Decoder with skip connections)
Loss Function: Binary Cross Entropy with Dice Loss
Optimizer: Adam
Epochs: 3 (for faster training)
Input Size: 128x128
Output: Segmentation mask highlighting target regions

🚀 How to Run
1️⃣ Install dependencies

```
pip install -r requirements.txt
````
2️⃣ Train the model

Run the training notebook:
```
train.py
```
3️⃣ Test / Inference

Run:
```
inference.py
```
This will:

Load the trained model from checkpoints/unet_final.pth
Predict masks for input images
Display and save the results in an outputs/ folder


📈 Results

The model learns to accurately segment target regions in the images.

Example:
Input Image: im_0001.png
Predicted Mask: pred_0001.png

🧩 Answers for Theory Questions

(a) Handle class imbalance → Use weighted loss (Dice Loss or focal loss)
(b) Evaluate boundary accuracy → Use IoU or Boundary F1-score
(c) Two augmentations → Horizontal flip, Random rotation (to improve generalization)

👨‍💻 Author
Developed by Kirubakaran
Project type: Image Segmentation using U-Net
