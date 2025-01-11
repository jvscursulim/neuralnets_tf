from __future__ import annotations

import keras

import tensorflow as tf


class CNN(keras.Model):
    """

    Args:
        keras (_type_): _description_
    """

    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        filters: list,
        kernel_size: int,
        strides: tuple,
        padding: str,
        activation: str,
        use_bias: bool,
        loss_func: str,
        maxpooling: bool = False,
        batch_normalization: bool = False,
        dropout: bool = False,
        dropout_rate: float = 0.3,
        *args,
        **kwargs,
    ) -> CNN:
        """_summary_

        Args:
            input_shape (tuple): Image shape.
            num_classes (int): The number of classes
            in the dataset.
            filters (list): A list with the number of
            filters in the convolutional layers.
            kernel_size (int): Kernel size.
            strides (tuple):
            padding (str): _description_
            activation (str): _description_
            use_bias (bool): _description_
            loss_func (str): _description_
            maxpooling (bool, optional): _description_. Defaults to False.
            batch_normalization (bool, optional): _description_. Defaults to False.
            dropout (bool, optional): _description_. Defaults to False.
            dropout_rate (float, optional): _description_. Defaults to 0.3.

        Returns:
            CNN: _description_
        """

        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.cnn_net = self._build_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            maxpooling=maxpooling,
            batch_normalization=batch_normalization,
            dropout=dropout,
            dropout_rate=dropout_rate,
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def _build_cnn(
        self,
        input_shape: tuple,
        num_classes: int,
        filters: list,
        kernel_size: int,
        strides: tuple,
        padding: str,
        activation: str,
        use_bias: bool,
        maxpooling: bool,
        batch_normalization: bool,
        dropout: bool,
        dropout_rate: float,
    ) -> keras.Model:
        """_summary_

        Args:
            input_shape (tuple): _description_
            num_classes (int): _description_
            filters (list): _description_
            kernel_size (int): _description_
            strides (tuple): _description_
            padding (str): _description_
            activation (str): _description_
            use_bias (bool): _description_
            maxpooling (bool): _description_
            batch_normalization (bool): _description_
            dropout (bool): _description_
            dropout_rate (float): _description_

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
            if maxpooling:
                x = keras.layers.MaxPooling2D()(x)
            if batch_normalization:
                x = keras.layers.BatchNormalization()(x)

        if dropout:
            x = keras.layers.Dropout(rate=dropout_rate)(x)
        x = keras.layers.Flatten()(x)
        if self.loss_func == "categorical_crossentropy":
            dense_activation = "softmax"
        elif self.loss_func == "binary_crossentropy":
            dense_activation = "sigmoid"
        x = keras.layers.Dense(
            units=num_classes, activation=dense_activation, use_bias=use_bias
        )(x)
        cnn_net = keras.Model(inputs=input_layer, outputs=x, name="CNN")

        return cnn_net

    @property
    def metrics(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """

        return [self.loss_tracker]

    def call(self, x) -> tf.Tensor:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            tf.Tensor: _description_
        """

        y_pred = self.cnn_net(x)

        return y_pred

    def train_step(self, data) -> dict:
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            dict: _description_
        """

        with tf.GradientTape() as tape:
            y_pred = self.cnn_net(data[0])
            if self.loss_func == "binary_crossentropy":
                loss = keras.losses.binary_crossentropy(y_true=data[1], y_pred=y_pred)
            elif self.loss_func == "categorical_crossentropy":
                loss = keras.losses.categorical_crossentropy(
                    y_true=data[1], y_pred=y_pred
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

        y_pred = self.cnn_net(data[0])
        if self.loss_func == "binary_crossentropy":
            loss = keras.losses.binary_crossentropy(y_true=data[1], y_pred=y_pred)
        elif self.loss_func == "categorical_crossentropy":
            loss = keras.losses.categorical_crossentropy(y_true=data[1], y_pred=y_pred)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}
