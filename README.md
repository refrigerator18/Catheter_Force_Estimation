# Catheter Force Estimation
Relevant code for our paper, ["Mathematical modeling and vision-based force estimation in a planar catheter using deep learning"](https://huzaifazar.me/catheter_force_estimation.pdf)

## Code Breakdown:

  The goal of this paper was to use machine learning techniques to predict the applied forces on a simulated cathter. The code for this project is split into two major sections, image processing via OpenCV, and subsequently the machine learning process via TensorFlow.

### 1. Image Processing

We have dataset of various simulated catheters with forces applied. The goal with image processing is to extract the center line of these catheter, in terms of real world distanes. Using these distances, we can formulate a mathematical model as described in the paper.


![final_61b96a6ab6df2f009b829b00_531529](https://user-images.githubusercontent.com/57844356/146122418-edb37209-3fcb-45bd-85a6-e249e2bba124.png)
