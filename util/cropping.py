import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

IMG_TYPES = 'mask'
LABEL_IMG = True
DIR_IMG_ORG = '/Users/alfonsocaro/Documents/Maestria/NUTI/AI/heatlh_mask/'

contenido = os.listdir(DIR_IMG_ORG)
count_file = 0

nombre_archivo = []
estado_archivo = []

for fichero in contenido:
    print(fichero)

    if os.path.isfile(os.path.join(DIR_IMG_ORG, fichero)) and fichero.endswith('.jpg'):
        try:
            count_file = count_file +1
            name_image = ('imagen_{0}_{1}.jpg').format(count_file, IMG_TYPES)

            img      = cv2.imread(os.path.join(DIR_IMG_ORG, fichero), cv2.IMREAD_COLOR)
            img_RGB  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_RGB2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_cascade = cv2.CascadeClassifier('/Users/alfonsocaro/Documents/Maestria/NUTI/AI/Cropping_Notebook/haarcascade_frontalface_default.xml')

            cropped_faces = []
            faces = face_cascade.detectMultiScale(img_RGB)

            p     = 10

            for (x,y,w,h) in faces:

                img = cv2.rectangle(img_RGB,(x,y),(x+w,y+h),(0,255,0),5)
                
                cropped_img = img_RGB2[y-p+1:y+h+p, x-p+1:x+w+p]
                cropped_faces.append(cropped_img)

            try:
                cropped_resized0 = cv2.resize(cropped_faces[0], (160, 160), interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(('/Users/alfonsocaro/Documents/Maestria/NUTI/AI/TallerAI/proccess_iamges/{0}/{1}').format(IMG_TYPES, name_image),  cv2.cvtColor(cropped_resized0, cv2.COLOR_RGB2BGR))

                nombre_archivo.append(name_image)
                estado_archivo.append(1)
            except Exception:
                pass 
        except Exception:
            pass

imagenes_dataset = list(zip(nombre_archivo,estado_archivo))

df = pd.DataFrame(data=imagenes_dataset, columns=["Nombre_Archivo","Estado"])

df.to_csv("/Users/alfonsocaro/Documents/Maestria/NUTI/AI/TallerAI/imagenes_mask.csv", index=False)