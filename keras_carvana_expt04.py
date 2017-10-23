# -*- coding: utf-8 -*-
"""
$Id$

@author: lambert.wixson
"""
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator #, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, concatenate, Dropout, Flatten, Dense, Input, Lambda, Merge 
from keras.optimizers import SGD
#%%
def setup_base_layers(inputs):
    lay1 = Conv2D(9, kernel_size=(3,3), padding = "valid", use_bias = True, trainable=True)(inputs) 
    lay3 = Activation('relu')(lay1)
    return lay3
#%%
    
if False:
    setup_prep()
    inputs = Input(shape=(160,240,3))
    lay3 = setup_base_layers(inputs)
    lay = setup_conv_layers1(lay3)
    lay = Flatten()(lay)
    lay = Dense(32)(lay)
    lay = Activation('relu')(lay)
    lay = Dropout(.20)(lay)
    lay = Dense(4, activation='softmax')(lay)
    modf = Model(inputs=inputs, outputs=lay)
    modf.summary()
    modf.layers[1].set_weights([rgb_concat_filts])
#%%  
    projdir = base + "\\datasets\\kaggle_carvana\\project_front_vs_back"
    batch_size=30   # We chose 30 because it evenly divides 6000, which is the number of training and test images we have
    train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=.100, height_shift_range=.100)
    test_datagen = ImageDataGenerator(rescale=1/255)
#%%
    train_generator = train_datagen.flow_from_directory(projdir + "\\train-color", target_size=(160,240), color_mode='rgb', 
                                                    classes=['yaw01','yaw05','yaw09','yaw13'], class_mode='categorical',  
                                                    batch_size=batch_size, shuffle=True, seed=193890)
                                                    #save_to_dir= projdir + "\\train\\augmented", save_prefix="aug")
    train_items = len(train_generator.filenames)
#%%
    validation_generator = test_datagen.flow_from_directory(projdir + "\\validate-color", target_size=(160,240), 
                                                            color_mode='rgb',
                                                            classes=['yaw01','yaw05', 'yaw09', 'yaw13'], class_mode='categorical',
                                                            batch_size=batch_size, shuffle=False)
    val_items = len(validation_generator.filenames)
    test_generator = test_datagen.flow_from_directory(projdir + "\\test-color", target_size=(160,240), 
                                                      color_mode='rgb',
                                                        classes=['yaw01','yaw05','yaw09','yaw13'], class_mode='categorical',
                                                        batch_size=batch_size, shuffle=False)
    test_items = len(test_generator.filenames)
    #%%  Build the model
    modf.compile(optimizer='sgd', loss='categorical_crossentropy', 
             metrics =['categorical_crossentropy', 'categorical_accuracy'] )
    callbacks = [EarlyStopping(monitor='val_categorical_accuracy', patience=5, verbose=1, mode='max'),
                 ModelCheckpoint('expt04_epoch{epoch:02d}.h5', monitor='val_categorical_accuracy', verbose=1, 
                                 save_best_only=True, save_weights_only=True)]
    modf.fit_generator(train_generator, steps_per_epoch = np.ceil(train_items/batch_size), epochs=50,
                   validation_data=validation_generator, validation_steps=np.ceil(val_items/batch_size),
                   callbacks=callbacks)
    # modf.save_weights(base + "/code/expt02_epoch50.h5")
    # modf.load_weights('./expt02_epoch50.h5')
    
    modf.compile(optimizer=SGD(.002), loss='categorical_crossentropy', 
             metrics =['categorical_crossentropy', 'categorical_accuracy'] )
    modf.compile(optimizer=SGD(.0004), loss='categorical_crossentropy', 
             metrics =['categorical_crossentropy', 'categorical_accuracy'] )