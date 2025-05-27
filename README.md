<div align="center">
        <h1>Medical Image Segmentation</h1>
            <p>This problem is tackled using a U-Net architecture</p>
            <p>
            <a href="https://github.com/VuThanhDat14122004/Image_analysis_and_processing/graphs/contributors">
                <img src="https://img.shields.io/github/contributors/VuThanhDat14122004/Image_analysis_and_processing" alt="Contributors" />
            </a>
            <a href="">
                <img src="https://img.shields.io/github/last-commit/VuThanhDat14122004/Image_analysis_and_processing" alt="last update" />
            <a href="https://github.com/VuThanhDat14122004/Image_analysis_and_processing/network/members">
		        <img src="https://img.shields.io/github/forks/VuThanhDat14122004/Image_analysis_and_processing" alt="forks" />
	        </a>
	        <a href="https://github.com/VuThanhDat14122004/Image_analysis_and_processing/stargazers">
		        <img src="https://img.shields.io/github/stars/VuThanhDat14122004/Image_analysis_and_processing" alt="stars" />
	        </a>
</div>

## Description
This project was developed for a Kaggle competition on medical image segmentation. It uses a U-Net model implemented in model.py, with custom loss functions in loss.py and a data pipeline defined in dataset.py. The training process is handled in train.py, and main.py serves as the entry point for training and testing

## Data
The dataset is divided into Train and Test sets.
- The Train set contains two folders:

    - Image/: 1087 grayscale medical images

    - Mask/: corresponding segmentation masks for each image

- The Test set includes 192 images in the Image/ folder. The masks are hidden and used for testing on Kaggle when submitting
## Hyperparameters
- Batch size = 16
- Optimizer: Adam with initial learning rate is 0.0001
- Loss function: The loss function combines BCELoss and Dice Loss, with a weighting factor of 0.5 assigned to each
- A learning rate scheduler is used, which reduces the learning rate by a factor of 0.5 if the validation loss does not improve for 3 consecutive epochs
- Early stopping is applied to halt training if the validation loss does not improve for 10 consecutive epochs
## Result
- Best validation loss is approximately 0.367 which achieved at epoch 23
![alt text](image.png)
- The final score on Kaggle is 0.8658 / 1 upon submission.