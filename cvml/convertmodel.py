import tensorflow as tf

# Convert the model
saved_model_dir = "C:/projects/fotobooth/cvml/hand-gesture-recognition-code/mp_hand_gesture"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
