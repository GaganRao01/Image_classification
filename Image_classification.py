import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
# These lines are importing some special tools that will help us work with images
# and use a pre-trained model called MobileNetV2.
model = MobileNetV2(weights='imagenet')
# This line loads the MobileNetV2 model that has been trained on a lot of images.
# The model can understand and recognize different objects in pictures
image_path = input("Enter the path of the image to classify: ")
# Prompt the user to enter the path of the image to classify
# below code is to Load and preprocess the image
img = Image.open(image_path)
img = img.convert('RGB')  # Ensure the image is in RGB format
img = img.resize((224, 224))  # Resize the image and line 15,16,17 These lines open the image you entered,
# make sure it is in a format the program can understand, and resize it to a specific size that the model expects.
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
# These 19,20,21 lines convert the image into an array of numbers that the model can analyze.
# Think of it as turning the image into a big list of numbers that the computer can understand.
predictions = model.predict(img_array)
decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=3)[0]
# These 24,25 lines use the model to make predictions about what is in the image.
#It looks at the numbers in the image array and tries to figure out what objects are in the picture.
# The top three predictions are then stored in the decoded_predictions variable.
# Display the results
print("Predictions:")
for pred in decoded_predictions:
    print(f"{pred[1]}: {pred[2] * 100:.2f}%")
# Finally, these lines display the predictions that the model made.
#  It prints out the objects it thinks are in the image,
#  along with a percentage that represents how confident it is about each prediction.