import model
import get_data
import tensorflow as tf
import os

if __name__ == '__main__':
    #path_to_dataset = "/Users/daoud.kadoch/Documents/counterfeit-detection-with-cnn/train/"
    path_to_dataset = "/home/ubuntu/counterfeit-detection-with-cnn/train/"
    x_train_1, x_train_2, y_train = get_data.get_siamese_data(path_to_dataset)

    model = model.get_SiameseModel()

    optimizer = tf.keras.optimizers.Adam(lr=3e-4)
    model.compile(loss="binary_crossentropy", metrics="accuracy", optimizer=optimizer)

    model_check = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("models_saved_best/"),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    model.fit(
        x=[x_train_1, x_train_2], y=y_train, batch_size=32, epochs=1000, verbose=1,
        validation_split=0.2, shuffle=True, callbacks=[model_check]
    )