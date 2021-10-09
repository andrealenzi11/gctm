from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.keras.constraints import MinMaxNorm
from tqdm import tqdm

tf.keras.backend.set_floatx("float64")
tf.executing_eagerly()


class RandomGenerator:
    """
        Neural Network that generates new synthetic data
    """

    def __init__(self,
                 gen_output_size: int,
                 **kwargs):
        """
            Initialization of the Generator Layers
        """
        self.gen_output_size = gen_output_size

    def generate_normal_randomness(self,
                                   mean: float,
                                   std: float,
                                   batch_size: int):
        """
            Method for generate the startup noise input tensor of the generator
        """
        return np.random.normal(loc=mean, scale=std, size=(batch_size, self.gen_output_size))
        # return np.random.uniform(0, 1, size=(batch_size, random_noise_size))


class AutoEncoder(tf.keras.Model):
    """
        Neural Network (Auto-Encoder) for data compression
    """

    def __init__(self,
                 discr_encoder_input_size: int,
                 discr_encoder_hidden1_size: int,
                 discr_encoder_output_size: int,
                 discr_decoder_input_size: int,
                 discr_decoder_hidden1_size: int,
                 discr_decoder_output_size: int,
                 **kwargs):
        """
            Initialization of the Discriminator Layers
        """
        super().__init__(name='discriminator', **kwargs)
        min_value = 0.
        max_value = 1.
        regularization_norm = "l1"

        # ==================== ENCODER ==================== #
        self.encoder_input = tf.keras.layers.InputLayer(input_shape=(discr_encoder_input_size,))  # not indispensable

        # self.encoder_noise = tf.keras.layers.GaussianNoise(stddev=discr_noise_std)
        self.encoder_input_dropout = tf.keras.layers.Dropout(rate=0.4)
        self.encoder_hidden1 = tf.keras.layers.Dense(
            units=discr_encoder_hidden1_size,
            activation=tf.nn.relu,
            kernel_regularizer=regularization_norm,
            activity_regularizer=regularization_norm,
            # kernel_constraint=MinMaxNorm(min_value=min_value, max_value=max_value)
        )
        self.encoder_hidden1_dropout = tf.keras.layers.Dropout(rate=0.1)
        self.encoder_output = tf.keras.layers.Dense(
            units=discr_encoder_output_size,
            activation=tf.nn.relu,
            kernel_regularizer=regularization_norm,
            activity_regularizer=regularization_norm,
            # kernel_constraint=MinMaxNorm(min_value=min_value, max_value=max_value)
        )

        # ==================== DECODER ==================== #
        self.decoder_input = tf.keras.layers.Input(shape=(discr_decoder_input_size,))  # not indispensable
        self.decoder_hidden1 = tf.keras.layers.Dense(
            units=discr_decoder_hidden1_size,
            activation=tf.nn.relu,
            kernel_regularizer=regularization_norm,
            activity_regularizer=regularization_norm,
            # kernel_constraint=MinMaxNorm(min_value=min_value, max_value=max_value)
        )
        self.decoder_hidden1_dropout = tf.keras.layers.Dropout(rate=0.1)
        self.decoder_output = tf.keras.layers.Dense(
            units=discr_decoder_output_size,
            activation=tf.nn.relu,
            kernel_regularizer=regularization_norm,
            activity_regularizer=regularization_norm,
            kernel_constraint=MinMaxNorm(min_value=min_value, max_value=max_value)
        )
        self.decoder_output_dropout = tf.keras.layers.Dropout(rate=0.4)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "encoder_input": self.encoder_input,
            "encoder_input_dropout": self.encoder_input_dropout,
            "encoder_hidden1": self.encoder_hidden1,
            "encoder_hidden1_dropout": self.encoder_hidden1_dropout,
            "encoder_output": self.encoder_output,
            "decoder_input": self.decoder_input,
            "decoder_hidden1": self.decoder_hidden1,
            "decoder_hidden1_dropout": self.decoder_hidden1_dropout,
            "decoder_output": self.decoder_output,
            "decoder_output_dropout": self.decoder_output_dropout,
        })
        return config

    def encode(self, inputs):
        """
            Encoding: from input samples to compressed latent spaces vectors
        """
        x = self.encoder_input_dropout(inputs)
        x = self.encoder_hidden1(x)
        x = self.encoder_hidden1_dropout(x)
        x = self.encoder_output(x)
        return x

    def decode(self, inputs):
        """
            Decoding: from compressed latent spaces vectors to reconstructed samples
        """
        x = self.decoder_hidden1(inputs)
        x = self.decoder_hidden1_dropout(x)
        x = self.decoder_output(x)
        x = self.decoder_output_dropout(x)
        return x

    def call(self, inputs, **kwargs):
        """
            Encoding and Decoding
        """
        latent_representation = self.encode(inputs)
        outputs = self.decode(latent_representation)
        return outputs


class RandomnessTextCompressionNN(tf.keras.Model):
    """
        Generative Cooperative Text Compression Neural Network:
        a custom network for dimensional reduction of textual documents
        based on the concepts of GAN and auto-encoder
    """

    def __init__(self,
                 num_epoches: int,
                 batch_size: int,
                 gen_output_size: int,
                 discr_learning_rate: float,
                 discr_encoder_input_size: int,
                 discr_encoder_hidden1_size: int,
                 discr_encoder_output_size: int,
                 discr_decoder_input_size: int,
                 discr_decoder_hidden1_size: int,
                 discr_decoder_output_size: int,
                 **kwargs):
        super().__init__(name="gctc_ablation", **kwargs)
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.gen_output_size = gen_output_size
        self.discr_learning_rate = discr_learning_rate
        self.discr_encoder_input_size = discr_encoder_input_size
        self.discr_encoder_hidden1_size = discr_encoder_hidden1_size
        self.discr_encoder_output_size = discr_encoder_output_size
        self.discr_decoder_input_size = discr_decoder_input_size
        self.discr_decoder_hidden1_size = discr_decoder_hidden1_size
        self.discr_decoder_output_size = discr_decoder_output_size
        self.discriminator_optimizer = tf.optimizers.Adam(learning_rate=discr_learning_rate, beta_1=0.5)
        self.is_trained = False
        self.generator = RandomGenerator(gen_output_size=self.gen_output_size)
        self.discriminator = None
        self.is_built = False
        self.discriminator_loss = tf.keras.losses.CosineSimilarity()
        self.real_min_v = None
        self.real_mean_v = None
        self.real_max_v = None
        self.fake_min_v = None
        self.fake_mean_v = None
        self.fake_max_v = None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_epoches": self.num_epoches,
            "batch_size": self.batch_size,
            "discr_learning_rate": self.discr_learning_rate,
            "gen_output_size": self.gen_output_size,
            "discr_encoder_input_size": self.discr_encoder_input_size,
            "discr_encoder_hidden1_size": self.discr_encoder_hidden1_size,
            "discr_encoder_output_size": self.discr_encoder_output_size,
            "discr_decoder_input_size": self.discr_decoder_input_size,
            "discr_decoder_hidden1_size": self.discr_decoder_hidden1_size,
            "discr_decoder_output_size": self.discr_decoder_output_size,
            "discriminator_optimizer": self.discriminator_optimizer,
            "is_trained": self.is_trained,
            "initial_random_noise": self.initial_random_noise,
            "generator": self.generator,
            "discriminator": self.discriminator,
            "is_built": self.is_built,
            "current_loss": self.current_loss,
        })
        return config

    def _set_item_size(self, num_features: int):
        self.gen_output_size = num_features
        self.discr_encoder_input_size = num_features
        self.discr_decoder_output_size = num_features

    def _check_built_status(self):
        if not self.is_built:
            raise Exception("The network has not yet been built!")

    def build_network(self, num_features: int):
        self._set_item_size(num_features)
        self.generator = RandomGenerator(gen_output_size=self.gen_output_size)
        self.discriminator = AutoEncoder(discr_encoder_input_size=self.discr_encoder_input_size,
                                         discr_encoder_hidden1_size=self.discr_encoder_hidden1_size,
                                         discr_encoder_output_size=self.discr_encoder_output_size,
                                         discr_decoder_input_size=self.discr_decoder_input_size,
                                         discr_decoder_hidden1_size=self.discr_decoder_hidden1_size,
                                         discr_decoder_output_size=self.discr_decoder_output_size)
        self.is_built = True

    def call(self, inputs, **kwargs):
        outputs = self.discriminator(inputs)
        return outputs

    @staticmethod
    def compute_mean_and_std(real_data: EagerTensor):
        mean_vector = tf.math.reduce_mean(real_data, axis=1).numpy()
        std_vector = np.sqrt(tf.math.reduce_variance(real_data, axis=1).numpy())
        return np.mean(mean_vector), np.mean(std_vector)

    def compute_discriminator_loss(self,
                                   real_input: EagerTensor,
                                   real_output: EagerTensor,
                                   generated_input: EagerTensor,
                                   fake_output: EagerTensor):
        real_loss = self.discriminator_loss(real_input, real_output)
        fake_loss = self.discriminator_loss(generated_input, fake_output)
        return real_loss + fake_loss

    def training_step(self,
                      batch: EagerTensor,
                      batch_size: int):
        self._check_built_status()
        with tf.GradientTape() as disc_tape:

            batch_mean, batch_std = self.compute_mean_and_std(real_data=batch)
            # print("batch mean and std:", batch_mean, batch_std)
            generated_batch = self.generator.generate_normal_randomness(mean=batch_mean,
                                                                        std=batch_std,
                                                                        batch_size=batch_size)
            generated_batch = tf.convert_to_tensor(value=generated_batch,
                                                   dtype=tf.float64)
            # print("batch:", batch.shape, batch.dtype)
            # print("generated_batch", generated_batch.shape, generated_batch.dtype)

            # === Calculate 'Real Output' / 'Fake Output' tensors for the discriminator === #
            real_discr_output = self.discriminator(batch)  # training=True
            self.real_min_v = tf.reduce_min(batch)
            self.real_mean_v = tf.reduce_mean(batch)
            self.real_max_v = tf.reduce_max(batch)
            fake_discr_output = self.discriminator(generated_batch)  # training=True
            self.fake_min_v = tf.reduce_min(generated_batch)
            self.fake_mean_v = tf.reduce_mean(generated_batch)
            self.fake_max_v = tf.reduce_max(generated_batch)

            # === Compute Cost Functions Losses === #
            discriminator_loss = self.compute_discriminator_loss(real_input=batch,
                                                                 real_output=real_discr_output,
                                                                 generated_input=generated_batch,
                                                                 fake_output=fake_discr_output)

            # === Calculate Gradients === #
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

            # === Apply Gradients === #
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                             self.discriminator.trainable_variables))
            return discriminator_loss

    def train(self, dataset):
        self._check_built_status()
        # Iterate over the epochs
        for i, epoch in enumerate(range(self.num_epoches)):
            print(f">>>>> EPOCH {i + 1}")
            # Iterate over the batches
            epoch_discriminator_losses = []
            for x in tqdm(iterable=range(0, len(dataset), self.batch_size), position=0, leave=True):
                batch = np.array(dataset[x: x + self.batch_size, :])
                batch = tf.convert_to_tensor(batch, dtype=tf.float64)
                discriminator_loss = self.training_step(batch=batch, batch_size=self.batch_size)
                epoch_discriminator_losses.append(discriminator_loss.numpy())
            print(f"\t\t epoch mean autoencoder loss:  {np.mean(epoch_discriminator_losses)} "
                  f" (std={np.std(epoch_discriminator_losses)})")
        self.is_trained = True
        print("\n values stats (min, mean, max):")
        print("\t - Real sample:")
        print("\t\t", self.real_min_v.numpy(), self.real_mean_v.numpy(), self.real_max_v.numpy())
        print("\t - Fake sample:")
        print("\t\t", self.fake_min_v.numpy(), self.fake_mean_v.numpy(), self.fake_max_v.numpy())

    def get_latent_space(self,
                         x_new: Sequence[str],
                         apply_softmax: bool = False) -> np.ndarray:
        """ Getting the compressed Latent Space of the new data given in input """
        self._check_built_status()
        if not self.is_trained:
            raise Exception("The auto-encoder is not trained!")
        encoded_latent_space = self.discriminator.encode(x_new)
        if apply_softmax:
            return tf.keras.activations.softmax(encoded_latent_space, axis=-1).numpy()
        else:
            return encoded_latent_space.numpy()
