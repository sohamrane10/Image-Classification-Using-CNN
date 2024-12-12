# -*- coding: utf-8 -*-
"""ImageClassificationUsingCNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gUBqM6zKJz9zCe4mMRV_Q2STLIifCvaR
"""

import tensorflow as tf
import tensorflow.keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np # Import numpy for array operations

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Use the 'map' function to apply the division to each element of the dataset
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img)
ds_test = ds_test.map(normalize_img)

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Convert the dataset into numpy arrays
train_images, train_labels = [], []
for image, label in ds_train:
    train_images.append(image.numpy())  # Convert image tensor to NumPy array
    train_labels.append(label.numpy())  # Convert label tensor to NumPy array

test_images, test_labels = [], []
for image, label in ds_test:
    test_images.append(image.numpy())  # Convert image tensor to NumPy array
    test_labels.append(label.numpy())  # Convert label tensor to NumPy array

test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap = plt.cm.binary)
  plt.xlabel(class_name[train_labels[i]]) # Access the label directly
plt.show()

# Step 6: Building the CNN model (customised model).
from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'), # Fixed typo: 'acivation' to 'activation'
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Step 7: Printing the model summary.
model.summary()

# Step 8: Compiling the CNN model.
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

# Step 9: Training the CNN model.
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Step 10: Evaluating the performance of the CNN model.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\n Test accuracy is: {test_acc}')

# Step 11: Plotting the training and validation accuracy and loss values.
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.grid()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,1])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.grid()

plt.show()