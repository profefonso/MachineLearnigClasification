import cv2 as cv2
import numpy as np
from tqdm import tqdm
from os import scandir, getcwd
from os.path import abspath


class LecturaImagenes():

    PATH_CON_MASK_IMG = 'images/ENTRENAMIENTO/CON_MASCARA'
    PATH_SIN_MASK_IMG = 'images/ENTRENAMIENTO/SIN_MASCARA'

    def cargaImagenes(self, hogd=False):

        # Matriz de descriptores
        imgConMascara = np.array([])
        imgSinMascara = np.array([])

        # Lectura de Imagenes Con Mascara
        print('Cargando Imagenes con mascara....')
        img_validas, imgConMascara= self.recorreFolder(self.PATH_CON_MASK_IMG, hogd)
        print("Se cargaron {0} imagenes con Mascara".format(img_validas))
        
        # vector de etiquetas
        etiquetasConMascara = np.ones(img_validas)

        # Lectura de Imagenes Sin Mascara
        print('Cargando Imagenes sin mascara....')
        img_validas, imgSinMascara = self.recorreFolder(self.PATH_SIN_MASK_IMG, hogd)
        print("Se cargaron {0} imagenes sin Mascara".format(img_validas))

        # Vector de etiquetas
        etiquetasSinMascara = np.zeros(img_validas)

        # Matriz de imagenes con caracteristicas
        matrizImagenes = np.concatenate((imgConMascara, imgSinMascara), axis=0)
        # Vector de etiquetas
        etiquetas = np.concatenate((etiquetasConMascara, etiquetasSinMascara), axis=0)

        return matrizImagenes, etiquetas


    def recorreFolder(self, folder_path, hogd):
        arreglo_imagenes = np.array([])
        img_validas = 0

        # Lectura de Imagenes con Mascara
        imgList = [abspath(imgs.path) for imgs in scandir(folder_path) if imgs.is_file()]

        for i in tqdm(range(len(imgList))):
            file = imgList[i]
            if file.endswith('.jpg'):
                img = cv2.imread(file, cv2.IMREAD_COLOR)

                if hogd:
                    # Se calcula el HOG
                    hog = cv2.HOGDescriptor()
                    h = hog.compute(img)
                    h2 = h.ravel()

                    arreglo_imagenes = np.hstack((arreglo_imagenes, h2))
                else:
                    arreglo_imagenes = np.hstack(np.asarray(img))

                img_validas = img_validas + 1

        if hogd:
            arreglo_imagenes = arreglo_imagenes.reshape((img_validas, len(h2)))
        
        return img_validas, arreglo_imagenes

    def test_model_image(self, imagen, clasificador):
        img = cv2.imread(imagen, cv2.IMREAD_COLOR)
        hog = cv2.HOGDescriptor()
        h = hog.compute(img)
        h2 = h.reshape((1, -1))
        
        return(clasificador.predict(h2))