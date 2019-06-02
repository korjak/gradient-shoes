from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
import numpy as np
import pandas as pd

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model


def train(model, train_generator, batch_size, validation_generator):
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
    model.save_weights('third_try.h5')


def generators(img_width, img_height, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    '''
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    '''

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(img_width, img_height),
        # classes=['main', 'front', 'side', 'back', 'top', 'bottom', 'crop'],
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(img_width, img_height),
        # classes=['main', 'front', 'side', 'back', 'top', 'bottom', 'crop'],
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator


def make_prediction(model, img_width, img_height):
    model.load_weights('third_try.h5')
    images = []
    img_name = []
    for i in os.listdir('data/test'):
        if not i.startswith('.'):
            img_name.append(i)
            img = image.load_img('data/test/{}'.format(i), target_size=(img_width, img_height), color_mode='grayscale')
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            images.append(img)

    # stack up images list to pass for prediction
    images = np.vstack(images)
    classes = model.predict_classes(images, batch_size=10)
    print(len(classes))
    print(len(np.array(img_name)))
    to_save = np.stack((np.array(img_name), classes), axis=1)
    #np.savetxt('predicted.csv', to_save)
    pd.DataFrame(to_save, columns=['image', 'class']).to_csv('predicted2.csv', index=False)


if __name__ == '__main__':
    img_width = 200
    img_height = 200
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)
    batch_size = 16

    model = create_model(input_shape)
    #train_generator, validation_generator = generators(img_width, img_height, batch_size)
    #train(model, train_generator, batch_size, validation_generator)
    # model.summary()
    #model.save_weights('second_try.h5')
    make_prediction(model, img_width, img_height)


