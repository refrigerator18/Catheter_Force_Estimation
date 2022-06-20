# Catheter Force Estimation
Relevant code for our paper, ["Mathematical modeling and vision-based force estimation in a planar catheter using deep learning"](https://huzaifazar.me/catheter_force_estimation.pdf). A sample dataset is provided.

**The dataset used in the paper makes use of more complex curves.*

## Code Breakdown:

  The goal of this paper was to use machine learning techniques to predict the applied forces on a simulated catheter. The code for this project is split into three major sections, image processing via OpenCV, training via RandomForest, and training via Nueral Netwroks.

### 1. Image Processing

We have dataset of various simulated catheters with forces applied. The goal with image processing is to extract the center line of these catheter, in terms of real world distanes. Using these distances, we can formulate an exponential mathematical model for each cathter, as described in the paper. 


![final_61b96a6ab6df2f009b829b00_531529](https://user-images.githubusercontent.com/57844356/146122418-edb37209-3fcb-45bd-85a6-e249e2bba124.png)


### 2. Training using RandomForest and Neural Network

 Using the mathematical model data (Processed_Data.csv), we can now start training. The algorithms are optimized as seen in the paper (Sections IV, part E and F). Using RandomForest, we obtain a Mean Average Error (MAE) of 1.36N using the data given. Using a Neural Network architecture, we obtain 0.52N MAE. 
 
 
