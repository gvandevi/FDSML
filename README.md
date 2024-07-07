# README

This project contains 5 notebooks and a python file with help functions.

1. **DataPreprocessing.ipynb**: 
   - Purpose: Preprocessing and storing data from the dataset available at [Kaggle Traffic Signs Classification](https://www.kaggle.com/code/valentynsichkar/traffic-signs-classification-with-cnn/).
   - Output: Data is saved in pickle files for easy loading in other notebooks.

2. **CNN.ipynb**:
   - Purpose: Training the original Convolutional Neural Network model.
   - Output: The trained model is saved as `model-3x3.keras`.

3. **Naïve_Trojan.ipynb**:
   - Purpose: Performing a naive trojan attack and checking its impact on the model.

4. **SPC.ipynb**:
   - Purpose: Checking scale consistency on example images to assess if it might be a method for trojan detection.

5. **GCAM.ipynb**:
   - Purpose: Gradient-weighted Class Activation Mapping : providing explanations for predictions regarding clean dataset & backdoor dataset 

6. **helpers.py**:
   - Description: Contains functions for data formatting, image processing and dataset creations. 