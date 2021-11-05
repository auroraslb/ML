import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Extract data
X = np.load('Xtrain_Classification_Part1.npy')
y = np.load('Ytrain_Classification_Part1.npy')

X_train, X_validate, train_labels, test_labels = train_test_split(X,y)

# Reshape to 50x50 pixel images
pixels = 50
train_images = []
test_images = []

for image in X_train:
    train_images.append(image.reshape(pixels, pixels))
train_images = np.array(train_images)

for image in X_validate:
    test_images.append(image.reshape(pixels, pixels))
test_images = np.array(test_images)

# Normalizing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building CNN Model
model = keras.Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))
model.summary()

model.compile(  optimizer='adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

# Train model
train_images = np.reshape(train_images, (len(train_images), 50, 50, 1))
test_images = np.reshape(test_images, (len(test_images),50, 50, 1))
history = model.fit(train_images, train_labels, validation_data = (test_images, test_labels), epochs = 10)

# Get testing data:
X_test = np.load('Xtest_Classification_Part1.npy')
X_test_images = []
for image in X_test:
    X_test_images.append(image.reshape(pixels, pixels))
X_test_images = np.array(X_test_images)
X_test_images = np.reshape(X_test_images, (len(X_test_images), 50, 50, 1))


predictions = model.predict(X_test_images)

# Turn predictions into labels
y_predicted = []

for i in predictions:
    if i[0]> i[1]:
        y_predicted.append(0)
    else:
        y_predicted.append(1)

y_predicted = np.array(y_predicted)

# Save predictions to file
np.save('predictions.npy', y_predicted)
