import tensorflow as tf
import training.model as models
import numpy as np
import cv2
import os
import pandas as pd
import tensorflow as tf


def generate_prediction_test(path_to_dataset:str, model:tf.keras.Model):
    """ Génére un fichier csv contenant les prédictions du modèle sur les données de test
        Params:
            path_to_dataset (str): chemin menant au dataset de test
            model (tf.keras.Model): modèle avec ses poids chargés
        Return:
            None
    """

    df = pd.DataFrame()
    id = []
    prediction = []

    nb_images = len([name for name in os.listdir(path_to_dataset + "img/")])
    cpt_img = 0
    directory = os.fsencode(path_to_dataset + "img/")
    for img_filename in os.listdir(directory):
        img_filename = img_filename.decode('utf-8')

        img_user = cv2.imread(path_to_dataset + "img/" + img_filename)
        img_user = np.array(cv2.resize(img_user, (100, 100))).reshape((1, 100, 100, 3))/255.0

        img_scan = cv2.imread(path_to_dataset + "scan/" + img_filename)
        img_scan = np.array(cv2.resize(img_scan, (100, 100))).reshape((1, 100, 100, 3))/255.0

        pred = model.predict([img_scan, img_user])[0][0]

        id.append(img_filename.split(".")[0])
        if pred < 0.5:
            prediction.append(0)
        else:
            prediction.append(1)

        cpt_img += 1
        print(str(cpt_img) + " / " + str(nb_images))

    df["id"] = id
    df["prediction"] = prediction
    df.to_csv("predictions.csv", index=False) # création du csv de prédiction des données de test


if __name__ == '__main__':

    path_to_model = "/training/models_saved_best/"
    model = models.get_SiameseModel()
    model.built = True
    model.load_weights(path_to_model) # chargement des poids du modèle

    path_to_dataset = "/Users/daoud.kadoch/Documents/counterfeit-detection-with-cnn/test/"

    generate_prediction_test(path_to_dataset, model)