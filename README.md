MNIST Classification using CNN

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits (0-9) using a dataset structured similarly to MNIST. The model is trained using PyTorch and includes data augmentation with Albumentations.

📌 Features

Dataset Handling: Loads and preprocesses images from a structured dataset.

Data Augmentation: Uses Albumentations for transformations.

Model Architecture: A simple CNN with two convolutional layers and batch normalization.

Training & Validation: Implements training and evaluation loops with accuracy and loss tracking.

Evaluation Metrics: Computes accuracy and displays a confusion matrix.

📂 Dataset Structure

Your dataset should be structured as follows:

data/
│── training/
│   ├── 0/  (Images of digit 0)
│   ├── 1/
│   ├── ...
│   ├── 9/
│── testing/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   ├── 9/

🛠 Installation

Ensure you have Python installed, then install dependencies:

pip install torch torchvision numpy pandas matplotlib seaborn albumentations scikit-learn tqdm opencv-python

🚀 Usage

Run the training script:

python train.py

Training & Validation

The script trains the CNN for 25 epochs and prints the loss for each epoch.

Evaluation

After training, the script computes the accuracy and generates a confusion matrix.

📊 Results

The model achieves high accuracy on digit classification.

The confusion matrix provides insight into misclassified digits.

💾 Model Saving & Loading

The trained model weights are saved as:

model_weights.pth

To load and use the trained model:

import torch
model = torch.load("model.pth")
model.eval()

🔥 Future Improvements

Implement more advanced CNN architectures (ResNet, EfficientNet).

Fine-tune hyperparameters for better accuracy.

Train on additional datasets to improve generalization.
