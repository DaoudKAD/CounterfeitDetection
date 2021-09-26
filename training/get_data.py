import numpy as np
import cv2
import os
import random


def get_siamese_data(path_to_dataset):

    x_train_1 = []
    x_train_2 = []
    y_train = []

    directory = os.fsencode(path_to_dataset + "img/")

    nb_images = len([name for name in os.listdir(path_to_dataset+"img/")])
    cpt_img = 0

    for img_filename in os.listdir(directory):
        img_filename = img_filename.decode('utf-8')
        img_user = cv2.imread(path_to_dataset + "img/" + img_filename)
        img_user = cv2.resize(img_user, (100, 100))
        img_user = np.array(img_user) / 255.0

        img_scan = cv2.imread(path_to_dataset + "scan/" + img_filename)
        img_scan = cv2.resize(img_scan, (100, 100))
        img_scan = np.array(img_scan)/255.0

        x_train_1.append(img_user)
        x_train_2.append(img_scan)
        y_train.append(1)

        cpt_img += 1
        print(str(cpt_img) + " / " + str(nb_images*2))

    for img_filename in os.listdir(directory):
        img_filename = img_filename.decode('utf-8')
        img_user = cv2.imread(path_to_dataset + "img/" + img_filename)
        img_user = cv2.resize(img_user, (100, 100))
        img_user = np.array(img_user) / 255.0

        id_img_scan = random.randint(0, nb_images - 1)
        img_scan = cv2.imread(path_to_dataset + "scan/" + str(id_img_scan) + ".jpg")
        img_scan = cv2.resize(img_scan, (100, 100))
        img_scan = np.array(img_scan) / 255.0

        x_train_1.append(img_user)
        x_train_2.append(img_scan)
        y_train.append(0)

        cpt_img += 1

        print(str(cpt_img)+" / "+str(nb_images*2))

    return np.array(x_train_1), np.array(x_train_2), np.array(y_train)

"""
if __name__ == '__main__':
    path_to_dataset = "/Users/daoud.kadoch/Documents/counterfeit-detection-with-cnn/train/"
    x_train_1, x_train_2, y_train = get_siamese_data(path_to_dataset)
"""