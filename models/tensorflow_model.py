import numpy
from tensorflow import keras


class TensorflowModel(keras.Model):
    def predict(self, batch, batch_idx) -> numpy.ndarray:
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        # ToDo: implement this method
