from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.backend import all
from keras import regularizers

classifier = Sequential()
classifier.add(Convolution2D(filters=44,kernel_size=(3,3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

classifier.add(Convolution2D(filters=64,kernel_size=(3,3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))


classifier.add(Convolution2D(filters=74,kernel_size=(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
classifier.add(Dropout(0.2))

classifier.add(Convolution2D(filters=84,kernel_size=(3,3), activation='relu'))
classifier.add(Dropout(0.2))

classifier.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
classifier.add(Dropout(0.2))


classifier.add(Flatten())

classifier.add(Dense(output_dim=512, activation='relu'))


classifier.add(Dense(output_dim=2, activation='softmax'))

classifier.compile(optimizer='sgd',
                   loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
import os

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.1,
    zoom_range=0.,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    "D:\\Study Material\\Mini Project\\dataset2-master\\images\\tr",
    target_size=(64,64),
    batch_size=8,
    shuffle=True,
    color_mode='grayscale',
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    "D:\\Study Material\\Mini Project\\dataset2-master\\images\\ts",
    target_size=(64,64),
    batch_size=8,
    shuffle=True,
    color_mode='grayscale',
    class_mode='categorical')

Output:
	Found 7843 images belonging to 2 classes.
	Found 4802 images belonging to 2 classes.


from IPython.display import display
from PIL import Image
classifier.fit_generator(
    training_set,
    steps_per_epoch=1000,
    epochs=20,
    validation_data=test_set,
    validation_steps=200)
    
    
import keras.utils
scores= classifier.evaluate_generator(test_set,steps=1000)
print(scores)
classifier.save('model.h5')

import numpy as np
from keras.preprocessing import image
test_image= image.load_img('eos. (4).jpeg',target_size=(64,64),color_mode='grayscale')
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict_proba(test_image)
training_set.class_indices

print(result)
