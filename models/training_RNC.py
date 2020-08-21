'''
CLASIFICADOR DE IMAGENES USANDO REDES NEURONALES CONVOLUCIONALES:
    * Creacion de Red Neuronal
    * Entrenamiento de la red
    * Generacion de Modelo y Pesos sinapticos

Maestria en Informatica 2020
'''
import os
import sys
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D

K.clear_session()

PATH_ENTRENA = 'images/ENTRENAMIENTO'
PATH_VALIDACION = 'images/VALIDACION'
PATH_MODEL = 'save_model'

# PARAMETROS REDES
ITERACION = 20
ALTURA, LONGITUD = 160, 160
BACH_IMAGES = 32
PASOS = 1000
PASOS_VALIDACION = 200
FILTROS_CONVOL_1 = 32
FILTROS_CONVOL_2 = 64
FILTRO_1_TAMANO = (3, 3)
FILTRO_2_TAMANO = (2, 2)
POOL_TAMANO = (2, 2)
CLASES = 2
LR = 0.0005

# PRE-PROCESAMIENTO DE IMAGENES
entrenamiento_datagen = ImageDataGenerator (
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

validacion_datagen = ImageDataGenerator (
    rescale=1./255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    PATH_ENTRENA,
    target_size=(ALTURA, LONGITUD),
    batch_size=BACH_IMAGES,
    class_mode='categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    PATH_VALIDACION,
    target_size=(ALTURA, LONGITUD),
    batch_size=BACH_IMAGES,
    class_mode='categorical'
)

# CREA LA RED CONVOLUCIONAL
# CAPAS DE LA RED
cnn = Sequential()
cnn.add(Convolution2D(
    FILTROS_CONVOL_1, 
    FILTRO_1_TAMANO,
    padding='same',
    input_shape=(ALTURA, LONGITUD, 3),
    activation='relu'
))

cnn.add(MaxPooling2D(pool_size=POOL_TAMANO))

cnn.add(Convolution2D(
    FILTROS_CONVOL_2, 
    FILTRO_2_TAMANO,
    padding='same',
    activation='relu'
))

cnn.add(MaxPooling2D(pool_size=POOL_TAMANO))
cnn.add(Flatten())
cnn.add(Dense(255, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(CLASES, activation='softmax'))
opt = Adam(lr=0.01)
cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# ENTRENAMIENTO DEL ALGORITMO
cnn.fit(
    imagen_entrenamiento, 
    steps_per_epoch=PASOS, 
    epochs=ITERACION, 
    validation_data=imagen_validacion, 
    validation_steps=PASOS_VALIDACION
)

if not os.path.exists(PATH_MODEL):
    os.mkdir(PATH_MODEL)

cnn.save(os.path.join(PATH_MODEL,'model_cnn.h5'))
cnn.save_weights(os.path.join(PATH_MODEL, 'weigths_model_cnn.h5'))