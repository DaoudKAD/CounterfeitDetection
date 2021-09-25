import tensorflow as tf
from keras.regularizers import l2
from keras import backend as K

def get_SiameseModel():

    # input pour le scan et pour l'image
    scan_input = tf.keras.layers.Input((100, 100, 3))
    image_input = tf.keras.layers.Input((100, 100, 3))

    # CNN
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (10, 10), activation='relu', input_shape=(100, 100, 3), kernel_regularizer=l2(2e-4)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3)))

    # images encodées par le CNN
    scan_encoded = model(scan_input)
    image_encoded = model(image_input)

    # différence entre les vecteurs encodés
    L1_layer = tf.keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([scan_encoded, image_encoded])

    # score de similarité
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L1_distance)

    # assemblage du modèle
    siamese_model = tf.keras.Model(inputs=[scan_input, image_input], outputs=prediction)

    return siamese_model
