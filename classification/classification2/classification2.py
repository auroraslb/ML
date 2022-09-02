import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import class_weight

# Extract data
X_train = np.load('Xtrain_Classification_Part2.npy')
train_labels = np.load('Ytrain_Classification_Part2.npy')
X_test = np.load('Xtest_Classification_Part2.npy')

# Reshape to 50x50 pixel images
pixels = 50
train_images = []
test_images = []

for image in X_train:
    train_images.append(image.reshape(pixels, pixels))

for image in X_test:
    test_images.append(image.reshape(pixels, pixels))

train_images = np.array(train_images)
test_images = np.array(test_images)


# Normalizing
train_images = train_images / 255.0
test_images = test_images / 255.0


# Sort data based on ethnicity
caucasian_nb_total = 0
african_nb_total = 0
asian_nb_total = 0
indian_nb_total = 0

caucasian_train = []
african_train = []
asian_train = []
indian_train = []

for index, label in enumerate(train_labels):
    if label == 0:
        caucasian_nb_total += 1
        caucasian_train.append(train_images[index])
    elif label == 1:
        african_nb_total += 1
        african_train.append(train_images[index])
    elif label == 2:
        asian_nb_total += 1
        asian_train.append(train_images[index])
    elif label == 3:
        indian_nb_total += 1
        indian_train.append(train_images[index])


# Augment caucasian data by adding mirrored images
caucasian_train = np.array(caucasian_train)
caucasian_augmented = []

caucasian_flipped = caucasian_train[:,:,::-1]

for index, image in enumerate(caucasian_flipped):
    caucasian_augmented.append(caucasian_train[index])
    caucasian_augmented.append(image)

caucasian_augmented = np.array(caucasian_augmented)


# Augment african data by adding mirrored images
african_train = np.array(african_train)
african_augmented = []

african_flipped = african_train[:,:,::-1]

for index, image in enumerate(african_flipped):
    african_augmented.append(african_train[index])
    african_augmented.append(image)

african_augmented = np.array(african_augmented)


# Augment asian data by adding mirrored images
asian_train = np.array(asian_train)
asian_augmented = []

asian_flipped = asian_train[:,:,::-1]

for index, image in enumerate(asian_flipped):
    asian_augmented.append(asian_train[index])
    asian_augmented.append(image)

asian_augmented = np.array(asian_augmented)


# Augment indian data by adding mirrored images
indian_train = np.array(indian_train)
indian_augmented = []

indian_flipped = indian_train[:,:,::-1]

for index, image in enumerate(indian_flipped):
    indian_augmented.append(indian_train[index])
    indian_augmented.append(image)

indian_augmented = np.array(indian_augmented)


# Update training set with augmented data
new_train_images = []
new_train_labels = []

count_caucasian = 0
count_african = 0
count_asian = 0
count_indian = 0

for index, label in enumerate(train_labels):
    if label == 0:
        new_train_images.append(caucasian_augmented[count_caucasian].reshape(pixels,pixels))
        new_train_labels.append(label)
        new_train_images.append(caucasian_augmented[count_caucasian + caucasian_nb_total].reshape(pixels,pixels))
        new_train_labels.append(label)
        count_caucasian += 1
    elif label == 1:
        new_train_images.append(african_augmented[count_african].reshape(pixels,pixels))
        new_train_labels.append(label)
        new_train_images.append(african_augmented[count_african + african_nb_total].reshape(pixels,pixels))
        new_train_labels.append(label)
        count_african += 1
    elif label == 2:
        new_train_images.append(asian_augmented[count_asian].reshape(pixels,pixels))
        new_train_labels.append(label)
        new_train_images.append(asian_augmented[count_asian + asian_nb_total].reshape(pixels,pixels))
        new_train_labels.append(label)
        count_asian += 1
    elif label == 3:
        new_train_images.append(indian_augmented[count_indian].reshape(pixels,pixels))
        new_train_labels.append(label)
        new_train_images.append(indian_augmented[count_indian + indian_nb_total].reshape(pixels,pixels))
        new_train_labels.append(label)
        count_indian += 1

new_train_images = np.array(new_train_images)


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
model.add(Dense(4, activation="softmax"))


# Compile model
model.compile(  optimizer='adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])


# Calculate weights
class_weights = class_weight.compute_class_weight('balanced', np.unique(new_train_labels), new_train_labels)
classes = [0, 1, 2, 3]
dict_weights = dict(zip(classes, class_weights.T))


# Train model
new_train_images = np.reshape(new_train_images, (len(new_train_images), 50, 50, 1))
new_train_labels = np.asarray(new_train_labels)

history = model.fit(new_train_images, new_train_labels, epochs = 10, class_weight=dict_weights)


# Predict for test set
test_images = np.reshape(test_images, (len(test_images),50, 50, 1))
predictions = model.predict(test_images)


# Turn predictions into labels
y_predicted = []

for i in predictions:
    y_predicted.append(np.argmax(i))

y_predicted = np.array(y_predicted)


# Write predictions to npy-file
np.save('predictions.npy', y_predicted)
