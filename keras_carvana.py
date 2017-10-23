# -*- coding: utf-8 -*-
"""
Initial experiments with keras.

$Id$
@author: lambert.wixson
"""
from os import path
from pathlib import Path
from skimage import color, filters, io, transform

import keras
from keras.preprocessing.image import ImageDataGenerator #, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.models import Model, Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, concatenate, Dropout, Flatten, Dense, Input, Lambda, Merge 
#%% 
def invert_dict(d):
    return dict([(v,k) for k,v in d.items()])

#%%
def find_mistakes(predicts, test_gen, show=False):
    """
    Figure out where the classifier made prediction mistakes.
    
    @param predicts: The predictions produced by test_gen
    """
    assert len(predicts) == len(test_gen.filenames)
    assert test_gen.shuffle == False
    mapClassnameToIndex = test_gen.class_indices
    mapIndexToClassname = invert_dict(mapClassnameToIndex)
    for (name, outputs) in zip(test_gen.filenames, predicts):
        parts = name.split('/')
        if len(parts) == 1:
            # If we get here, there was no forward slash in the filename
            parts = name.split('\\')
            if len(parts) == 1:
                assert False, "Could not extract class name from front of filenames"
        cls = parts[0]
        truedex = test_gen.class_indices[cls]
        maxdex = np.argmax(outputs)
        if truedex != maxdex:
            print("{0} mistaken for {1} - outputs {2}".format(name, mapIndexToClassname[maxdex], outputs))
            if show:
                newshow(test_gen.directory + "/" + name, winname = name)
    return
            
            
#%%
def setup_prep():
    global sobel_dx, sobel_dy, concat_filts, rgb_concat_filts
    sobel_dx = np.array(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)), dtype='float32') / 8
    sobel_dy = np.array(((-1, -2, -1), (0, 0, 0), (1, 2, 1)), dtype='float32') / 8
    
    # Make a dx and dy filter for each of the 3 bands.  Each of these is 3x3x3
    r_dx = np.stack((sobel_dx, np.zeros((3,3), dtype='float32'), np.zeros((3,3), dtype='float32')), axis=-1)
    r_dy = np.stack((sobel_dy, np.zeros((3,3), dtype='float32'), np.zeros((3,3), dtype='float32')), axis=-1)
    
    g_dx = np.stack((np.zeros((3,3), dtype='float32'), sobel_dx, np.zeros((3,3), dtype='float32')), axis=-1)
    g_dy = np.stack((np.zeros((3,3), dtype='float32'), sobel_dy, np.zeros((3,3), dtype='float32')), axis=-1)
    
    b_dx = np.stack((np.zeros((3,3), dtype='float32'), np.zeros((3,3), dtype='float32'), sobel_dx), axis=-1)
    b_dy = np.stack((np.zeros((3,3), dtype='float32'), np.zeros((3,3), dtype='float32'), sobel_dy), axis=-1)
    rgb_concat_filts = np.stack((r_dx, r_dy, g_dx, g_dy, b_dx, b_dy), axis=-1)
    
    # We now need to assemble the filters in the order that Keras expects, which 
    # seems to be row, col, input channel, output channel
    concat_filts = np.concatenate((sobel_dx.reshape((3,3,1,1)), sobel_dy.reshape((3,3,1,1))), axis=3)
#%%
rgb2gray = np.array((.2125, .7154, ))
setup_prep()
input_shape=(160, 240)
model = Sequential()

layer1 = Conv2D(2, kernel_size=(3,3), input_shape= (160, 240, 1), use_bias = False, trainable=False)
model.add(layer1)
layer1.set_weights([concat_filts])

layer2 = Lambda(lambda x: K.log(x))
model.add(layer2)
#%%
# Test it out - add another dimension at end to represent the single channel
img = seq[0]
inp = img.reshape(160,240,1)
# Add another dimension at the front to represent the sample 
batch = inp.reshape((1,)+ inp.shape)
out = model.predict(batch, batch_size=1, verbose=1)
f = newshow(out[0,:,:,0])     # See that the dx filter is working properly
f = newshow(out[0,:,:,1])     # See that the dy filter is working properly

out2 = model.predict(batch, batch_size=1, verbose=1)
inputs = Input(shape=(160,240,1))
#%%  Try it using the Functional API, so that I'll be able to skip layers
def setup_base_layers(inputs):
    lay1 = Conv2D(2, kernel_size=(3,3), use_bias = False, trainable=False)(inputs)
    lay2 = Lambda(K.square)(lay1)
    lay3 = keras.layers.concatenate([lay1, lay2])
    return lay3

#%%
lay3 = setup_base_layers(inputs)

modf = Model(inputs=inputs, outputs=lay3)
modf.layers[1].set_weights([concat_filts])
modf.summary()
outf = modf.predict(batch, batch_size=1, verbose=1)
f = newshow(outf[0,:,:,0], winname="dx")     # See that the dx filter is working properly
f = newshow(outf[0,:,:,1], winname="dy") 
f = newshow(outf[0,:,:,2], winname="dxsq")     # See that the dx filter is working properly
f = newshow(outf[0,:,:,3], winname="dysq") 

#%%
def setup_conv_layers1(lay3):
    lay = Conv2D(32, kernel_size=(3,3), dilation_rate=(1,1), padding='valid')(lay3)
    lay = Activation('relu')(lay)
    lay = MaxPooling2D(pool_size=(2,2), padding='valid')(lay)                    # Take image to 80 x 120
    lay = Conv2D(32, kernel_size=(3,3), dilation_rate=(1,1), padding='valid')(lay)
    lay = Activation('relu')(lay)
    lay = MaxPooling2D(pool_size=(2,2), padding='valid')(lay)                    # Take image to 40 x 60
    lay = Conv2D(32, kernel_size=(3,3), dilation_rate=(1,1), padding='valid')(lay)
    lay = Activation('relu')(lay)
    lay = MaxPooling2D(pool_size=(2,2), padding='valid')(lay)                    # Take image to 20 x 30
    lay = Conv2D(32, kernel_size=(3,3), dilation_rate=(1,1), padding='valid')(lay)
    lay = Activation('relu')(lay)
    return lay

#%%
def setup_conv_layers2(lay3):
    lay = Conv2D(32, kernel_size=(3,3), dilation_rate=(2,2))(lay3)
    lay = Activation('relu')(lay)
    lay = MaxPooling2D(pool_size=(2,2))(lay)                    # Take image to 80 x 120
    lay = Conv2D(32, kernel_size=(3,3), dilation_rate=(2,2))(lay)
    lay = Activation('relu')(lay)
    lay = MaxPooling2D(pool_size=(2,2))(lay)                    # Take image to 40 x 60
    lay = Conv2D(32, kernel_size=(3,3), dilation_rate=(2,2))(lay)
    lay = Activation('relu')(lay)
    lay = MaxPooling2D(pool_size=(2,2))(lay)                    # Take image to 20 x 30
    lay = Conv2D(32, kernel_size=(5,5), dilation_rate=(1,1))(lay)
    lay = Activation('relu')(lay)

#%%
lay = setup_conv_layers1(lay3)
lay = Flatten()(lay)
lay = Dense(32)(lay)
lay = Activation('relu')(lay)
lay = Dropout(.20)(lay)

lay = Dense(2, activation='softmax')(lay)


modf = Model(inputs=inputs, outputs=lay)
modf.summary()
#%%  Set up the data generators

projdir = base + "\\datasets\\kaggle_carvana\\project_front_vs_back"
batch_size=30   # We chose 30 because it evenly divides 6000, which is the number of training and test images we have
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=.100, height_shift_range=.100, channel_shift_range=.2)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(projdir + "\\train", target_size=(160,240), color_mode='grayscale', 
                                                    classes=['yaw01','yaw09'], class_mode='categorical',  
                                                    batch_size=batch_size, shuffle=True, seed=193890)
                                                    #save_to_dir= projdir + "\\train\\augmented", save_prefix="aug")
#%%
validation_generator = test_datagen.flow_from_directory(projdir + "\\validate", target_size=(160,240), color_mode='grayscale',
                                                        classes=['yaw01','yaw09'], class_mode='categorical',
                                                        batch_size=batch_size, shuffle=False)
val_items = len(validation_generator.filenames)

test_generator = test_datagen.flow_from_directory(projdir + "\\test", target_size=(160,240), color_mode='grayscale',
                                                        classes=['yaw01','yaw09'], class_mode='categorical',
                                                        batch_size=batch_size, shuffle=False)
#%%  Build the model
modf.compile(optimizer='sgd', loss='categorical_crossentropy', 
             metrics =['binary_crossentropy', 'categorical_crossentropy', 'categorical_accuracy'] )

modf.fit_generator(train_generator, steps_per_epoch = np.ceil(6000/batch_size), epochs=50,
                   validation_data=validation_generator, validation_steps=np.ceil(1144/batch_size))

modf.save_weights(base + "/code/first_try.h5")

#%% Check that the generators work the way I think they do

# The last batch will always have a smaller number in its batch.  Unless you forget to call reset() in which case
# the small batch could be anywhere, and the batches will not line up with the .filenames attribute. 
validation_generator.reset()
i = 0
for batch in validation_generator:
    #print(batch)
    print(len(batch[1]))
    i=i+1
    if i >= np.ceil(1144/batch_size):
        break
    
#%% Evaluate 
test_generator.reset()
modf.evaluate_generator(test_generator, np.ceil(test_items/batch_size))
Out[150]: 
[0.0081246644626418937,
 0.0081246641458860448,
 0.0081246644626418937,
 0.99682486631016043]

modf.metrics_names
Out[151]: 
['loss',
 'binary_crossentropy',
 'categorical_crossentropy',
 'categorical_accuracy']

test_generator.reset()
pred = modf.predict_generator(test_generator, np.ceil(test_items/batch_size), verbose=1)
find_mistakes(pred, test_generator)
find_mistakes(pred, test_generator, show=True)

validation_generator.reset()
valpred = modf.predict_generator(validation_generator, np.ceil(val_items/batch_size))
find_mistakes(valpred, validation_generator)
find_mistakes(valpred, validation_generator, show=True)
    
#%%
if False:
    f = newshow(seq[0])
    videofig(len(seq), redraw_fn, winname="raw", cmap='gray',
         proc_func=lambda f: seq[f])
    videofig(len(seq), redraw_fn, winname="gmag",
         proc_func=lambda f: makegmag(seq[f]))
    

