# gesture-classification-cnn

## 🧠 Project Overview

This project focuses on building a hand gesture classification system using
Convolutional Neural Networks (CNN).

The system takes hand gesture images as input and predicts the corresponding
gesture class using deep learning models such as VGG16, MobileNet, and ResNet.

The project includes data preparation, model training, evaluation, and testing
through a simple graphical user interface.

## 🎯 Objectives

- Classify hand gestures from images using deep learning
- Apply and compare different CNN architectures
- Improve model accuracy using transfer learning
- Evaluate model performance using training, validation, and test data

## 🧠 Models Used

The following pre-trained CNN models were used in this project:

### 1️⃣ VGG16
- Deep convolutional neural network with 16 layers
- Known for its simple and uniform architecture
- Used as a baseline model for gesture classification

### 2️⃣ MobileNet
- Lightweight and efficient CNN architecture
- Designed for mobile and low-resource devices
- Provides faster training and inference with good accuracy

### 3️⃣ ResNet
- Deep residual network that uses skip connections
- Solves the vanishing gradient problem
- Achieves high accuracy on complex image classification tasks

| Model         | Model Size | Speed     | Accuracy | Advantages                                          | Disadvantages                     |
| ------------- | ---------- | --------- | -------- | --------------------------------------------------- | --------------------------------- |
| **VGG16**     | Large      | Slow      | Good     | Simple architecture, stable training                | High memory usage, slow inference |
| **MobileNet** | Small      | Very Fast | Good     | Lightweight, efficient, suitable for mobile devices | Slightly lower accuracy           |
| **ResNet**    | Medium     | Medium    | High     | Deep network with skip connections, high accuracy   | More complex architecture         |


| Model       | Train Acc | Val Acc | Test Acc |
|------------|----------|---------|----------|
| VGG16      | 92%      | 88%     | 87%      |
| MobileNet  | 90%      | 89%     | 88%      |
| ResNet     | 95%      | 92%     | 91%      |

## 🗂️ Dataset

- The dataset consists of hand gesture images
- Each gesture is stored in a separate folder
- Images are resized and normalized before training

### Data Split
- Training set
- Validation set
- Test set

- Training: 80%
- Validation: 10%
- Test: 10%

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Jupyter Notebook


## 🖥️ Graphical User Interface (GUI)

A simple GUI was developed to:
- Upload a hand gesture image
- Predict the gesture class using the trained model
- Display the prediction result


## 🚀 How to Run the Project

1. Install the required libraries
```bash
pip install -r requirements.txt
2. Run the training notebooks inside the notebooks folder

3. Run the GUI application
```bash
python gui/app.py
