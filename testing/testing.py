import tensorflow as tf
import training.model as models
import numpy as np
import cv2


if __name__ == '__main__':

    path_to_model = "/Users/daoud.kadoch/PycharmProjects/CounterfeitDetection/training/models_saved/"
    model = models.get_SiameseModel()
    model.built = True
    model.load_weights(path_to_model)


    res = model([np.zeros((1,100, 100, 3)), np.zeros((1,100, 100, 3))])
    print(res)