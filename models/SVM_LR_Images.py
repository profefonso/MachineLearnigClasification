# Cargo las librerías necesarias
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from load_images import LecturaImagenes

TEST_IMG_SIN_MASCARA = 'images/TEST/imagen_798_faces.jpg'
TEST_IMG_CON_MASCARA = 'images/TEST/imagen_154_mask.jpg'

lectura_imagenes = LecturaImagenes()
matriz_imagenes, etiquetas = lectura_imagenes.cargaImagenes(hogd=False)

print(" ")

# CLASIFICADOR CON SVM
print('::::: CLASIFICADOR CON SVM :::::')

svm = SVC(kernel='linear')
scores = cross_val_score(svm, matriz_imagenes, etiquetas, cv=5)
print(scores)

svm_m = SVC(kernel="linear")
svm_m.fit(matriz_imagenes, etiquetas)
# TEST SIN MASCARA
res = lectura_imagenes.test_model_image(TEST_IMG_SIN_MASCARA, svm_m)
print("{0} fue clasificado como: {1}".format(TEST_IMG_SIN_MASCARA, res))

# TEST CON MASCARA
res = lectura_imagenes.test_model_image(TEST_IMG_CON_MASCARA, svm_m)
print("{0} fue clasificado como: {1}".format(TEST_IMG_CON_MASCARA, res))

print(" ")

# CLASIFICADOR REGRESION LOGISTICA
print('::::: CLASIFICADOR CON REGRESION LOGISTICA :::::')
logistic = LogisticRegression(random_state=0, max_iter=100000)
scores = cross_val_score(logistic, matriz_imagenes, etiquetas, cv=5)
print(scores)

logistic.fit(matriz_imagenes, etiquetas)
# TEST SIN MASCARA
res = lectura_imagenes.test_model_image(TEST_IMG_SIN_MASCARA, logistic)
print("{0} fue clasificado como: {1}".format(TEST_IMG_SIN_MASCARA, res))

# TEST CON MASCARA
res = lectura_imagenes.test_model_image(TEST_IMG_CON_MASCARA, logistic)
print("{0} fue clasificado como: {1}".format(TEST_IMG_CON_MASCARA, res))