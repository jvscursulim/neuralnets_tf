from __future__ import annotations

import keras

import numpy as np
import tensorflow as tf


class Autoencoder(keras.Model):
    """Autoencoder neural network
    class

    Args:
        keras (_type_): _description_
    """

    def __init__(
        self,
        input_shape: tuple,
        filters: list,
        kernel_size: int,
        strides: tuple,
        padding: str,
        activation: str,
        use_bias: bool,
    ) -> Autoencoder:
        """Initializes the Autoencoder class

        Args:
            input_shape (tuple): _description_
            batch_size (int): _description_
            filters (list): _description_
            kernel_size (int): _description_
            strides (tuple): _description_
            padding (str): _description_
            activation (str): _description_
            use_bias (bool): _description_

        Returns:
            Autoencoder: _description_
        """

        super().__init__()
        self.encoder = self._build_encoder(
            input_shape=input_shape,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
        )
        self.decoder = self._build_decoder(
            input_shape=input_shape,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    @staticmethod
    def _build_encoder(
        input_shape: tuple,
        filters: list,
        kernel_size: int,
        strides: tuple,
        padding: str,
        activation: str,
        use_bias: bool,
    ) -> keras.Model:
        """_summary_

        Args:
            input_shape (tuple): _description_
            filters (list): _description_
            kernel_size (int): _description_
            strides (tuple): _description_
            padding (str): _description_
            activation (str): _description_
            use_bias (bool): _description_

        Returns:
            keras.Model: _description_
        """

        input_layer = keras.layers.Input(shape=input_shape)
        for idx, num_filters in enumerate(filters):
            if idx == 0:
                x = keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    activation=activation,
                    use_bias=use_bias,
                )(input_layer)
            else:
                x = keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    activation=activation,
                    use_bias=use_bias,
                )(x)
        x = keras.layers.Conv2D(
            filters=input_shape[-1],
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
        )(x)

        encoder = keras.Model(inputs=input_layer, outputs=x, name="encoder")

        return encoder

    @staticmethod
    def _build_decoder(
        input_shape: tuple,
        filters: list,
        kernel_size: int,
        strides: tuple,
        padding: str,
        activation: str,
        use_bias: bool,
    ) -> keras.Model:
        """_summary_

        Args:
            input_shape (tuple): _description_
            filters (list): _description_
            kernel_size (int): _description_
            strides (tuple): _description_
            padding (str): _description_
            activation (str): _description_
            use_bias (bool): _description_

        Returns:
            keras.Model: _description_
        """

        input_layer = keras.layers.Input(
            shape=(
                int(np.ceil(input_shape[0] / 2 ** len(filters))),
                int(np.ceil(input_shape[1] / 2 ** len(filters))),
                input_shape[-1],
            )
        )
        for idx, num_filters in enumerate(filters[::-1]):
            if idx == 0:
                x = keras.layers.Conv2DTranspose(
                    filters=num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    activation=activation,
                    use_bias=use_bias,
                )(input_layer)
            else:
                x = keras.layers.Conv2DTranspose(
                    filters=num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    activation=activation,
                    use_bias=use_bias,
                )(x)

        x = keras.layers.Conv2D(
            filters=input_shape[-1],
            kernel_size=kernel_size,
            strides=(1, 1),
            padding=padding,
            activation="linear",
        )(x)
        decoder = keras.Model(inputs=input_layer, outputs=x, name="decoder")

        return decoder

    @property
    def metrics(self) -> list:

        return [self.loss_tracker]

    def call(self, x) -> tf.Tensor:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            tf.Tensor: _description_
        """

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def train_step(self, data) -> dict:
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            dict: _description_
        """

        with tf.GradientTape() as tape:
            encoded = self.encoder(data if not isinstance(data, tuple) else data[0])
            decoded = self.decoder(encoded)
            loss = keras.losses.mean_squared_error(
                y_true=data if not isinstance(data, tuple) else data[1], y_pred=decoded
            )

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data) -> dict:
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            dict: _description_
        """

        encoded = self.encoder(data if not isinstance(data, tuple) else data[0])
        decoded = self.decoder(encoded)
        loss = keras.losses.mean_squared_error(
            y_true=data if not isinstance(data, tuple) else data[1], y_pred=decoded
        )

        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}
