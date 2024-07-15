import tensorflow as tf

MODEL_PATH = 'Q:\projects\\apple_variety\AppleVariety\Pommelogy\AppleVariety\ml_model\\try_pommelogy.keras'

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
