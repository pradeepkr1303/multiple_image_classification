# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Convolution
# make 32 feature detectors with a size of 3x3
# choose the input-image's format to be 64x64 with 3 channels
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation="relu"))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(activation="relu", units=128))
# make sure the units is equal to the number of classes
classifier.add(Dense(activation="softmax", units=2))

# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# use ImageDataGenerator to preprocess the data
from keras.preprocessing.image import ImageDataGenerator

# augment the data that we have
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

# prepare training data
training_data = train_datagen.flow_from_directory('./training',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# prepare test data
test_data = test_datagen.flow_from_directory('./testing',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# finally start computation
classifier.fit_generator(training_data,
                         steps_per_epoch = (8000 / 320),
                         epochs = 1,
                         validation_data = test_data,
                         validation_steps = 20)

# to make predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('spider-mite1.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
# training_data.class_indices
if result[0][0] == 1:
    prediction = 'Healthy'
elif result[0][1] == 1:
    prediction = 'Spider-mite'
"""elif result[0][2] == 1:
    prediction = 'carrot'
elif result[0][3] == 1:
    prediction = 'cauliflower'
elif result[0][4] == 1:
    prediction = 'garlic'
elif result[0][5] == 1:
    prediction = 'onion'
elif result[0][6] == 1:
    prediction = 'pumpkin'
elif result[0][7] == 1:
    prediction = 'radish'
elif result[0][8] == 1:
    prediction = 'tomato'
elif result[0][9] == 1:
    prediction = 'watermelon'
elif result[0][10] == 1:
   prediction = 'unpredicted'
"""
print(prediction)
