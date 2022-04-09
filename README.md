# StructureUpsampling
Simple CNN for line and edge upsampling worked for [ZITS](https://github.com/DQiaole/ZITS_inpainting).

## Results
![](./asserts/SSU.png)


## Train

Note that only line(wireframe) is used as the training target. And the trained model can work for both line and edge(canny).

1. Preprocess and get pkl of lines(wireframes) from images with [LSM-HAWP](https://github.com/ewrfcas/MST_inpainting).

2. Setting image file lists in configs/config.yml

3. python trian.py
