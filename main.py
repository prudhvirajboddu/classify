import random, re, math
random.seed(a=42)
import numpy as np
# from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn

strategy=tf.distribute.get_strategy()


train_dir='./train'

height=width=224
depth=3
classes=3

CFG = dict(  
    LR_START          =   0.00005,
    LR_MAX            =   0.000020,
    LR_MIN            =   0.000001,
    LR_RAMPUP_EPOCHS  =   5,
    LR_SUSTAIN_EPOCHS =   0,
    LR_EXP_DECAY      =   0.8,
    epochs            =   30,   
)
# import keras_preprocessing
# from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

# Change Number of epochs and batch_size from here.(Used in model training)
NUM_EPOCHS = 35        # number of epochs
BS = 16              # bat  ch Size

image_generator = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=45,
      width_shift_range=0.4,
      height_shift_range=0.3,
      horizontal_flip=True,
      zoom_range=0.3,
      vertical_flip=True,
      fill_mode='constant',
      validation_split=0.1
      # add, remove or change parameters for better accuracy
	)

train_generator = image_generator.flow_from_directory(
	train_dir,
    subset="training",
	target_size=(height,width),
	class_mode='categorical',
	color_mode="rgb",
	shuffle=True,
	batch_size=BS
  # add, remove or change parameters for better accuracy
)

validation_generator = image_generator.flow_from_directory(
    train_dir,
    shuffle=False,
    target_size=(height,width), 
    subset="validation",
    class_mode='categorical',
	color_mode="rgb"
    # add, remove or change parameters for better accuracy
    )

def get_lr_callback(cfg):
    lr_start   = cfg['LR_START']
    lr_max     = cfg['LR_MAX'] * strategy.num_replicas_in_sync
    lr_min     = cfg['LR_MIN']
    lr_ramp_ep = cfg['LR_RAMPUP_EPOCHS']
    lr_sus_ep  = cfg['LR_SUSTAIN_EPOCHS']
    lr_decay   = cfg['LR_EXP_DECAY']
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    return lr_callback


es=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=4,
                                    mode='max',restore_best_weights=False,verbose=1)

def get_model(cfg):
    model_input = tf.keras.Input(shape=(224,224,3), name='imgIn')

    dummy = tf.keras.layers.Lambda(lambda x:x)(model_input)

    constructor = getattr(tf.keras.applications, f'InceptionResNetV2')
    x = constructor(include_top=False, weights='imagenet', 
                    input_shape=(height,width,depth),
                    pooling='avg')(dummy)
    x = tf.keras.layers.Dropout(0.2,seed=1024)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)
    outputs = [x]
        
    model = tf.keras.Model(model_input, outputs, name='aNetwork')
    model.summary()
    return model

def compile_new_model(cfg):    
    model = get_model(cfg)
        
    model.compile(
        optimizer = 'adam',
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy'])
        
    return model

model=compile_new_model(CFG)

# model.summary()

model.fit(train_generator,
          epochs=NUM_EPOCHS,
          callbacks=[get_lr_callback(CFG),es],
          validation_data=validation_generator,
        #   validation_steps=math.ceil(validation_generator.samples/validation_generator.batch_size),
          steps_per_epoch=math.ceil(train_generator.samples/train_generator.batch_size),
          verbose=1)  

# print(train_generator.samples)
# print(validation_generator.samples)
# print(train_generator.batch_size)
# print(validation_generator.batch_size)
