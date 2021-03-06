#!/usr/bin/env python3
"""
Create a class NST that performs tasks for neural style transfer
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
tf.enable_eager_execution()


class NST:
    """
    Class NST
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or \
           style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or \
           content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.model = model
        self.generate_features()
        self.gram_style_features = style_features
        self.content_feature = content_features

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its values are between 0 and 1
        and its largest side is 512 pixels
        Returns the scaled image
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or \
           image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        height = image.shape[0]
        width = image.shape[1]
        scale = 512 / max(height, width)
        new_shape = (int(scale * height), int(scale * width))
        image = np.expand_dims(image, axis=0)
        image_scaled = tf.clip_by_value(tf.image.resize_bicubic
                                        (image, new_shape) / 255.0, 0.0, 1.0)
        return image_scaled

    def load_model(self):
        """
        Creates the model used to calculate cost
        """
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False)
        x = vgg.input

        for layer in vgg.layers[1:]:
            layer.trainable = False
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(name=layer.name)(x)
            else:
                x = layer(x)
        global model
        model = tf.keras.models.Model(vgg.input, x, name="model")
        outputs = [model.get_layer(layer).get_output_at(1)
                   for layer in self.style_layers + [self.content_layer]]
        model = tf.keras.models.Model(vgg.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate gram matrices
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(
                input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, shape=[-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(tf.transpose(a), a) / tf.cast(n, tf.float32)
        gram = tf.reshape(gram, shape=[1, -1, channels])
        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        Sets the public instance attributes
        """
        global style_features
        global content_features
        st_lay_len = len(self.style_layers)
        style_image = preprocess_input(self.style_image*255)
        content_image = preprocess_input(self.content_image*255)
        features = model(style_image) + model(content_image)
        style_features = [self.gram_matrix(layer) for
                          layer in features[:st_lay_len]]
        content_features = features[-1:]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        Returns: the layer???s style cost
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or len(
                style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        if not isinstance(gram_target,  (tf.Tensor, tf.Variable)) or len(
                gram_target.shape) != 3 or gram_target.shape[0] != 1:
            raise TypeError(
                "gram_target must be a tensor of shape [1, {c}, {c}]")
        style_output = self.gram_matrix(style_output)
        return tf.reduce_mean((style_output-gram_target)**2)
