import tensorflow as tf
model = tf.keras.models.load_model("model/model_v1_cnn.h5")
model.save("model/model_v1.keras")
