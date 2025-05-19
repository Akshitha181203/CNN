# CNN-based Binary Image Classifier using TensorFlow
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to perform binary image classification. The model is trained on a custom dataset and achieves strong performance with accuracy reaching **88.89% on validation** and **80.00% on test data**.
---

## Dataset Structure

The dataset is organized in the following format:

Dataset_Lab/
├── Dataset/
│ ├── class_1/
│ └── class_2/
└── test/
├── class_1/
└── class_2/


- `Dataset/` — Contains the full dataset used for training and validation (240 images across 2 classes).
- `test/` — Contains 20 images used for evaluating the model after training.

---

## Tech Stack

- Python
- TensorFlow
- Keras
- Matplotlib
- Google Colab

---

## Model Architecture

The model is a sequential CNN with the following layers:

- **Conv2D + ReLU**: 16 filters
- **MaxPooling2D**
- **Conv2D + ReLU**: 32 filters
- **MaxPooling2D**
- **Conv2D + ReLU**: 64 filters
- **MaxPooling2D**
- **Flatten**
- **Dense**: 64 units + ReLU
- **Dropout**: 0.5 to reduce overfitting
- **Dense**: Output layer with 2 units (softmax for binary classification)

---

## Training Details

- **Image size**: 128x128
- **Batch size**: 32
- **Train/Validation Split**: 70/30
- **Epochs**: 25
- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `adam`

---

## Results

| Metric              | Value     |
|---------------------|-----------|
| Final Train Accuracy| 100.00%   |
| Validation Accuracy | 88.89%    |
| Test Accuracy       | 80.00%    |

---

### Training Graphs

Training and validation accuracy and loss over 25 epochs are plotted using `matplotlib`.

---

## Key Features

- Clean and modular code using `tf.data` API for efficient data loading and prefetching.
- Normalization using `Rescaling(1./255)` layer.
- Dropout layer to mitigate overfitting.
- Prefetching with `AUTOTUNE` for performance optimization.
- Visualization of training/validation curves.
- Evaluation on a separate test dataset.

---

## How to Run

1. Upload the dataset to your Google Drive under `Dataset_Lab/`
2. Mount Google Drive in Colab
3. Run the script to:
   - Load and split the dataset
   - Normalize and prefetch data
   - Build and train the CNN model
   - Plot accuracy/loss graphs
   - Evaluate on the test set

---

## Future Improvements

- Add data augmentation (e.g., rotation, flipping).
- Introduce early stopping and model checkpointing.
- Hyperparameter tuning (batch size, optimizer, learning rate).

---

## Contact

For any queries or feedback, feel free to reach out to:

**Peguda Akshitha**  
📧 [Email](akshithapeguda@gmail.com)  
🔗 [GitHub](https://github.com/Akshitha181203)
