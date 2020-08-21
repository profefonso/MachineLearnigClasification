import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

ALTURA, LONGITUD = 160, 160
MODELO = 'save_model/model_cnn.h5'
WEIGTHS = 'save_model/weigths_model_cnn.h5'

cnn = load_model(MODELO)
cnn.load_weights(WEIGTHS)

def predict(file):
    x = load_img(file, target_size=(LONGITUD, ALTURA))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arreglo = cnn.predict(x)
    print(arreglo)
    resp = arreglo[0]
    result = np.argmax(resp)

    if(result==0):
        print('CON MASCARA')
    else:
        print('SIN MASCARA')
    
    return result

predict('images/imagen_923_faces.jpg')
predict('images/900x600.jpg')
predict('images/foto_1.jpg')
predict('images/fgy.jpg')