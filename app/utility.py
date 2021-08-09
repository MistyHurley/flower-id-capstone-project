import tensorflow as tf
import numpy as np

saved_model = tf.keras.models.load_model('floweridentifymodel.h5')
class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']


class Utility:
    @staticmethod
    def get_model_prediction(filename):
        img = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        predictions = saved_model.predict(img_array)
        print(predictions)
        #score = tf.nn.softmax(predictions[0])
        #confidence = 100 * np.max(score)
        #confidence = predictions[0][np.argmax(score)] * 100

        return predictions
