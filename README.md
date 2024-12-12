## Image-Classification-Using-CNN
This project implements an image classification model using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images across 10 classes, with 50,000 images for training and 10,000 images for testing.

## Prerequisites
To run this project, ensure you have the following libraries installed:

- Python 3.8 or later
- TensorFlow
- TensorFlow Datasets (tfds)
- NumPy
- Matplotlib

## Dataset
The CIFAR-10 dataset is loaded using TensorFlow Datasets. It contains the following 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Data Preprocessing
- The images are normalized to have pixel values between 0 and 1.
- The dataset is converted into NumPy arrays for ease of use.

## Model Architecture
The CNN model is built using TensorFlow's Keras API. The architecture includes:

1. **Convolutional Layers**:
   - Three convolutional layers with 32, 64, and 64 filters respectively, each using ReLU activation and kernel size of (3,3).
2. **Pooling Layers**:
   - Two max-pooling layers with pool size (2,2).
3. **Flattening**:
   - The output of the convolutional layers is flattened into a 1D vector.
4. **Fully Connected Layers**:
   - A dense layer with 64 neurons and ReLU activation.
   - An output dense layer with 10 neurons (one for each class).

### Model Summary
The model summary is printed to provide an overview of the architecture.

## Model Compilation
The model is compiled using:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

## Training
The model is trained for 10 epochs using the training dataset, with validation performed on the testing dataset.

## Evaluation
The test dataset is used to evaluate the model's accuracy, and the results are printed.

## Results Visualization
The following plots are generated to visualize the training process:
1. Training and validation accuracy over epochs.
2. Training and validation loss over epochs.

## How to Run
1. Clone the repository.
2. Install the required libraries using pip:
   ```bash
   pip install tensorflow tensorflow-datasets matplotlib numpy
   ```
3. Run the script in a Python environment:
   ```bash
   python image_classification_cnn.py
   ```

## Output
- The training process outputs the accuracy and loss values for each epoch.
- A plot of training and validation accuracy/loss is displayed.
- The final test accuracy is printed.

## Conclusion
This project demonstrates the effectiveness of CNNs in image classification tasks using the CIFAR-10 dataset. With further tuning and augmentation techniques, the model's accuracy can be improved.

---

Feel free to customize and extend the model as needed!

