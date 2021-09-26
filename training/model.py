import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def get_SiameseModel():

    input_shape = (100, 100, 3)

    # input pour le scan et pour l'image
    scan_input = tf.keras.layers.Input(input_shape)
    image_input = tf.keras.layers.Input(input_shape)


    # 64 128 128 256
    # CNN
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(512, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3)))
    model.add(tf.keras.layers.Dense(1024, activation='sigmoid', kernel_regularizer=l2(1e-3)))

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
