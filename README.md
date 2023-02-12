# Neo6D

## 6D headpose estimation for video editing (Re-implementation for educational purposes) based on https://github.com/thohemp/6DRepNet

For use in After Effects (create_null.jsx)
<p align="center">
  <img src="demo.gif" alt="animated" />
</p>

Shamiko model by [@norm_sNs](https://twitter.com/norm_sNs)

## Step 0: Create new **Conda** environment

## Step 1: Install requirements
Note: Makes sure you have **CUDA** and installed **GPU** version of torch
```
matplotlib >= 3.3.4
numpy >= 1.19.5
opencv-python >= 4.5.5
pandas >= 1.1.5
Pillow >= 8.4.0
scipy >= 1.5.4
torch >= 1.10.1
torchvision >= 0.11.2

pip install git+https://github.com/elliottzheng/face-detection.git@master
```

## Step 1.5: Download pretrained models
https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_?usp=sharing

## Step 2: Run main.py
```
options:
  -h, --help            show this help message and exit
  --gpu GPU_ID          GPU device id to use (Default=0)
  --model MODEL_PATH    Path to model file
  --source SOURCE_PATH  Source video's path
  --save_video SAVE_VIDEO
                        Save video with visualization (Default=False)
  --save_csv SAVE_CSV   Save csv (Default=True)
  --cpu CPU             CPU only mode (Default=False)
```

Example:
```
python main.py --model ".\6DRepNet_300W_LP_AFLW2000.pth" --source ".\tylerSmall_0.mp4"
```

## Step 3: Run create_null.jsx in After Effects
Parent your 3D things' attributes to the corresponding attributes of the newly created null

Optional: Add smooth() to the null attributes to smooth the movements