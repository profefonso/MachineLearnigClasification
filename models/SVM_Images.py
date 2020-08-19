# Cargo las librerías necesarias
import cv2 as cv2
from sklearn.svm import SVC

from load_images import LecturaImagenes

TEST_IMG_SIN_MASCARA = 'images/TEST/imagen_798_faces.jpg'
TEST_IMG_CON_MASCARA = 'images/TEST/imagen_154_mask.jpg'

def test(imagen, clasificador):
    """
    Funcion para predecir el tipo de una muestra
    
    Parámetros:
    imagen --       ruta de la imagen a leer
    clasificador -- clasificador de sklearn ya entrenado
    
    Retorna:
    valor -- retorna la predicción realizada
    """
    img = cv2.imread(imagen, cv2.IMREAD_COLOR)
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    h2 = h.reshape((1, -1))
    return(clasificador.predict(h2))



# -----------------------------
# Ejecución de la prueba
# -----------------------------

# Obtengo los datos de trainning
lectura_imagenes = LecturaImagenes()
X_Train, y_train = lectura_imagenes.cargaImagenes()

# Creo una SVM con kernel linear
clf = SVC(kernel="linear")

# Entreno la SVM
clf.fit(X_Train, y_train)

# -------------------------------------------
# Pruebo con imágenes
# -------------------------------------------

# Ejemplo negativo
res = test(TEST_IMG_SIN_MASCARA, clf)
print(TEST_IMG_SIN_MASCARA, " fue clasificado como: ", res)

# Ejemplo positivo
res = test(TEST_IMG_CON_MASCARA, clf)
print(TEST_IMG_CON_MASCARA, " fue clasificado como: ", res)
