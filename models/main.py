# Cargo las librerías necesarias
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn import model_selection
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from load_images import LecturaImagenes

lectura_imagenes = LecturaImagenes()
matriz_imagenes, etiquetas = lectura_imagenes.cargaImagenes(hogd=True)
print(etiquetas)

# Division del Dataset en 80% train y 20% prueba
x_train, x_test, y_train, y_test = train_test_split(matriz_imagenes, etiquetas, 
test_size=0.20, random_state=0)
#print(x_test)
#print(y_test)

print('::::: CLASIFICADOR CON REGRESION LOGISTICA :::::')
# Entrenando el modelo
logisticRegr = LogisticRegression(random_state=0, max_iter=1000)
history = logisticRegr.fit(x_train, y_train)

# Realizamos prediccion con la informacion de test
#logisticRegr.predict(x_test[0].reshape(1,-1))
predictions = logisticRegr.predict(x_test)

# Usamos el Score para obtener el Accuracy del modelo
print('::::: ACCURACY :::::')
score = cross_val_score(logisticRegr, matriz_imagenes, etiquetas, cv=5)
text_score = "Accuracy Score = {:f} (+/- {:f})".format(-score.mean(), score.std())
print(text_score)

# Generamos la matriz de confusion
print('::::: MATRIZ CONFUSION :::::')
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.title('CONFUSION MATRIX - LOGISTIC REGRESSION')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.suptitle(text_score)
plt.show()
print(metrics.classification_report(y_test, predictions))

train_sizes = [10, 20, 30]
train_sizes, train_scores, valid_scores = model_selection.learning_curve(logisticRegr, matriz_imagenes, etiquetas, train_sizes = train_sizes, cv=5,scoring = 'neg_mean_squared_error')
print(train_sizes)
print(train_scores)
print(valid_scores)

print(" ")


# CLASIFICADOR CON SVM
print('::::: CLASIFICADOR CON SVM :::::')
# Entrenando el modelo
svm = SVC(kernel='linear', max_iter=1000)
history = svm.fit(x_train, y_train)

# Realizamos prediccion con la informacion de test
#logisticRegr.predict(x_test[0].reshape(1,-1))
predictions = svm.predict(x_test)

# Usamos el Score para obtener el Accuracy del modelo
print('::::: ACCURACY :::::')
score = cross_val_score(svm, matriz_imagenes, etiquetas, cv=5)
text_score = "Accuracy Score = {:f} (+/- {:f})".format(-score.mean(), score.std())
print(text_score)

# Generamos la matriz de confusion
print('::::: MATRIZ CONFUSION :::::')
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Greens_r')
plt.title('CONFUSION MATRIX - SVM')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.suptitle(text_score)
plt.show()
print(metrics.classification_report(y_test, predictions))


