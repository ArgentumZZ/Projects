# Dense residual networks in generative adversarial networks

# 1. Import libraries
# 2. Create a function that builds the generator (inverted CNN)
# 3. Create a function that builds the discriminator (CNN)
# 4. Create a function tha builds the generator-discriminator network (combined model)
# 5. Create a function to visualize the generated images and save them in the working directory
# 6. Create a function to load the dataset
# 7. Create a function to select randomly training samples
# 8. Create a function to generate fake images
#  9. Create a training function that executes the functions above
# 9.1 - Train the discriminator model only
# 9.2 - Freeze the weights of the discriminator and train the combined model (generator + discriminator)
# 9.3 - Execute the image_visualization function
# Maybe create Inception Score/Frechet Inception Distance
from keras.datasets import cifar10
from keras.layers import (Dense, Reshape, Flatten, Dropout, BatchNormalization,
                          Activation, Add, Conv2D, Conv2DTranspose)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras import Input, Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import time
# How to install TensorFlow: https://www.youtube.com/watch?v=hHWkvEcDBO0&t=288s&ab_channel=AladdinPersson

# Convolution -> Batch normalization -> Activation -> Dropout

# List of potential optimizers that can be used: https://keras.io/api/optimizers/
# List of potential loss functions that can be used: https://keras.io/api/losses/
# List of potential metrics that can be used: https://keras.io/api/metrics/
# Weight Initialization: https://keras.io/api/layers/initializers/
# Weight Regularizers: https://keras.io/api/layers/regularizers/
# Weight Constraints: https://keras.io/api/layers/constraints/
# How to load data from files: https://keras.io/api/data_loading/
# Callbacks: https://keras.io/api/callbacks/
# Model Check Point: Save the best model to a specific filepath by monitoring validation loss
# Early Stopping: If the validation loss doesn't improve after x number for patience parameter, stop
# Option 1: Padding with zeros = 'same' and stride=1 (output size is the same as the input size)
# Option 2: No Padding = 'valid (neuron's receptive field lies strictly within valid positions inside the input)
numpy.random.seed(1337)


def build_residual_discriminator(optimizer=tf.keras.optimizers.Adam(),
                                 d_dropout_rate=0.2,
                                 d_momentum=0.99,
                                 loss_function='binary_crossentropy',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=1337)):

    """Using residual connects in the discriminator. Every next layer takes the previous outputs (from all the layers).
    Each residual connection is down-sampled appropriately to match dimensions. Finally, all outputs are added (not
    concatenated)."""

    # 32x32x3 - x_0
    inputs = Input(shape=(32, 32, 3))

    # Block x_1, 32x32x3 -> 32x32x128
    residual_main_1 = inputs  # 32x32x3
    residual_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(residual_main_1)
    # residual_1 = BatchNormalization(momentum=d_momentum)(residual_1)
    x = Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding="same")(inputs)
    # x = BatchNormalization(momentum=d_momentum)(x)
    x = Add()([x, residual_1])
    x = LeakyReLU()(x)  # output -> 32x32x128
    # x = Dropout(rate=d_dropout_rate)(x)

    # Block x_2, 32x32x128 -> 16x16x128
    residual_2_main = x
    residual_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding="same")(residual_2_main)
    # residual_2 = BatchNormalization(momentum=d_momentum)(residual_2)
    x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    # x = BatchNormalization(momentum=d_momentum)(x)
    # residual_main_1 must be down-sampled to 16x16x128
    res = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same')(residual_main_1)
    # res = BatchNormalization(momentum=d_momentum)(res)
    x = Add()([x, residual_2, res])
    x = LeakyReLU()(x)  # output -> 16x16x128
    # x = Dropout(rate=d_dropout_rate)(x)

    # Block x_3, 16x16x128 -> 8x8x128
    residual_3_main = x
    residual_3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same')(residual_3_main)
    # residual_3 = BatchNormalization(momentum=d_momentum)(residual_3)
    x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    # x = BatchNormalization(momentum=d_momentum)(x)
    res = Conv2D(filters=128, kernel_size=(1, 1), strides=(4, 4), padding='same')(residual_main_1)
    # res = BatchNormalization(momentum=d_momentum)(res)
    res_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(4, 4), padding='same')(residual_2_main)
    # res_2 = BatchNormalization(momentum=d_momentum)(res_2)
    x = Add()([x, residual_3, res_2, res])
    x = LeakyReLU()(x)  # output -> 8x8x128
    # x = Dropout(rate=d_dropout_rate)(x)

    x = Flatten()(x)
    x = Dropout(rate=0.4)(x)
    x = Dense(units=1, activation='sigmoid')(x)

    res_discriminator = Model(inputs=inputs,  outputs=x)
    res_discriminator.summary()
    res_discriminator.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    return res_discriminator


def build_residual_generator(latent_dim=128,
                             g_momentum=0.99,
                             g_dropout_rate=0.2,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.02, seed=1337)):

    """Using residual connects in the generator. Every next layer takes the previous outputs (from all the layers).
    Each residual connection is down-sampled appropriately to match dimensions. Finally, all outputs are added (not
    concatenated)."""

    inputs = Input(shape=latent_dim)
    x = Dense(units=(8*8*128))(inputs)
    # x = BatchNormalization(momentum=g_momentum)(x)
    # x = LeakyReLU(alpha=0.2)(x)
    x_inputs = Reshape(target_shape=(8, 8, 128))(x)

    # Block x_1, 8x8x128 -> 16x16x128
    residual_main_1 = x_inputs
    residual_1 = Conv2DTranspose(filters=256, kernel_size=(1, 1), strides=(2, 2), padding="same")(residual_main_1)
    # residual_1 = BatchNormalization(momentum=g_momentum)(residual_1)
    x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same")(x_inputs)
    # x = BatchNormalization(momentum=g_momentum)(x)
    x = Add()([x, residual_1])
    x = LeakyReLU(alpha=0.2)(x)
    # x = Dropout(rate=g_dropout_rate)(x)

    # Block x_2, 16x16x128 -> 32x32x128
    residual_main_2 = x
    residual_2 = Conv2DTranspose(filters=256, kernel_size=(1, 1), strides=(2, 2), padding="same")(residual_main_2)
    # residual_2 = BatchNormalization(momentum=g_momentum)(residual_2)
    x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    # x = BatchNormalization(momentum=g_momentum)(x)
    res = Conv2DTranspose(filters=256, kernel_size=(1, 1), strides=(4, 4), padding="same")(residual_main_1)
    # res = BatchNormalization(momentum=g_momentum)(res)
    x = Add()([x, residual_2, res])
    x = LeakyReLU()(x)  # output -> 32x32x128
    # x = Dropout(rate=g_dropout_rate)(x)

    # Block x_3, 32x32x128 -> 32x32x3
    residual_main_3 = x
    residual_3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(residual_main_3)
    # residual_3 = BatchNormalization(momentum=g_momentum)(residual_3)
    x = Conv2D(filters=3, kernel_size=(5, 5), strides=(1, 1), padding="same")(x)
    # x = BatchNormalization(momentum=g_momentum)(x)
    res = Conv2DTranspose(filters=3, kernel_size=(1, 1), strides=(2, 2), padding='same')(residual_main_2)
    # res = BatchNormalization(momentum=g_momentum)(res)
    res_2 = Conv2DTranspose(filters=3, kernel_size=(1, 1), strides=(4, 4), padding='same')(residual_main_1)
    # res_2 = BatchNormalization(momentum=g_momentum)(res_2)
    x = Add()([x, residual_3, res, res_2])
    x = Activation('tanh')(x)  # output -> 32x32x3

    res_generator = Model(inputs=inputs,  outputs=x)
    res_generator.summary()

    return res_generator


def build_residual_gan(residual_discriminator, residual_generator, optimizer=tf.keras.optimizers.Adam()):

    """
    Pass the residual discriminator and residual generator models, freeze the weights of the res_discriminator,
    combine models and compile the combined network.
    """

    residual_discriminator.trainable = False
    combined_model = Sequential(name="COMBINED_MODEL")
    combined_model.add(residual_generator)
    combined_model.add(residual_discriminator)
    combined_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return combined_model


def build_discriminator(optimizer=tf.keras.optimizers.Adam(),
                        d_dropout_rate=0.2,
                        d_momentum=0.99,
                        loss_function='binary_crossentropy',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=1337)):

    """Avoid batch_norm at the input of the discriminator, it may lead to instability."""

    disc_model = Sequential(name="DISCRIMINATOR")

    # Increase depth: 32x32x128
    disc_model.add(Conv2D(filters=128, input_shape=(32, 32, 3), kernel_size=(3, 3), strides=(1, 1), padding="same"))
                          # kernel_initializer=kernel_initializer))
    # BN not recommended here
    # disc_model.add(BatchNormalization(momentum=d_momentum))
    disc_model.add(LeakyReLU(alpha=0.2))
    # disc_model.add(Dropout(rate=d_dropout_rate))

    # Down-sample: 16x16x128
    disc_model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
                          # kernel_initializer=kernel_initializer))
    # disc_model.add(BatchNormalization(momentum=d_momentum))
    disc_model.add(LeakyReLU(alpha=0.2))
    # disc_model.add(Dropout(rate=d_dropout_rate))

    # Down-sample: 8x8x128
    disc_model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
                          # kernel_initializer=kernel_initializer))
    # disc_model.add(BatchNormalization(momentum=d_momentum))
    disc_model.add(LeakyReLU(alpha=0.2))
    # disc_model.add(Dropout(rate=d_dropout_rate))

    # Flatten to 8192 nodes = 8x8x128
    disc_model.add(Flatten())
    disc_model.add(Dropout(rate=d_dropout_rate))
    disc_model.add(Dense(units=1, activation='sigmoid'))

    disc_model.summary()
    disc_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    return disc_model


def build_generator(latent_dim=128,
                    g_momentum=0.99,
                    g_dropout_rate=0.2,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.02, seed=1337)):

    """Avoid batch_norm at the output of the generator, it may lead to instability.
    Generate_fake_samples function accepts Gaussian noise to create images."""

    gen_model = Sequential(name="GENERATOR")

    # Number of nodes: 8192
    gen_model.add(Dense(units=8*8*128, input_dim=latent_dim))
    # gen_model.add(BatchNormalization(momentum=g_momentum))

    # Reshape to (8x8x128)
    gen_model.add(Reshape(target_shape=(8, 8, 128)))

    # Add parameters, still 8x8x128
    gen_model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding="same"))
                         # kernel_initializer=kernel_initializer, use_bias=False))
    # gen_model.add(BatchNormalization(momentum=g_dropout_rate))
    gen_model.add(LeakyReLU(alpha=0.2))

    # Upscale to 16x16x128
    gen_model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
                                  # kernel_initializer=kernel_initializer, use_bias=False))
    # gen_model.add(BatchNormalization(momentum=g_momentum))
    gen_model.add(LeakyReLU(alpha=0.2))
    # gen_model.add(Dropout(rate=g_dropout_rate))

    # upscale to 32x32x128
    gen_model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
                                  # kernel_initializer=kernel_initializer, use_bias=False))
    # gen_model.add(BatchNormalization(momentum=g_momentum))
    gen_model.add(LeakyReLU(alpha=0.2))
    # gen_model.add(Dropout(rate=g_dropout_rate))

    # Change depth to 32x32x3
    gen_model.add(Conv2D(filters=3, kernel_size=(5, 5), strides=(1, 1), padding="same"))
                         # kernel_initializer=kernel_initializer))
    gen_model.add(Activation("tanh"))
    gen_model.summary()

    """The model is not directly trained like the discriminator, so no need to compile it."""
    return gen_model


def build_gan_combined_network(discriminator, generator, optimizer=tf.keras.optimizers.Adam()):

    """
    Pass the discriminator and generator models, freeze the weights of the discriminator, combine models and compile
    the combined network.
    """

    discriminator.trainable = False
    combined_model = Sequential(name="COMBINED_MODEL")
    combined_model.add(generator)
    combined_model.add(discriminator)
    combined_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return combined_model


def plot_generated_images(epoch, generator, examples=100, latent_dim=100, dim=(10, 10), figsize=(10, 10), filepath=None):

    """
    After some training the generator parameters have been updated, so generate new images from some latent space,
    graph and observe how they look after each update. Generate an input (noise) with size (examples, latent_dim),
    feed it to the generator and generate 100 example images with size (32, 32, 3).
    """

    noise = np.random.normal(loc=0, scale=1, size=[examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 32, 32, 3)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow((generated_images[i]+1)/2, interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{filepath}_{epoch}.png')


def load_my_cifar10_data():

    """
    Load the cifar10 dataset, change to float type for faster computation and re-scale.
    """

    (x_train, _), (_, _) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 127.5 - 1
    return x_train


def select_real_samples(x_train, batch_size):

    """
    Generate "batch_size" number of random integers between min and max index of the dataset and select some samples
    from the real dataset.
    """

    indices = np.random.randint(low=0, high=x_train.shape[0], size=batch_size)
    real_images = x_train[indices]
    return real_images


def generate_fake_samples(generator, batch_size, latent_dim):

    """
    Generate latent points (noise) from the normal distribution and create fake images.
    """

    noise = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
    generated_fake_images = generator.predict(noise)
    return generated_fake_images


def smooth_real_labels(x, large_range=False, factor=0.3):

    """
    Recommended: Apply to real images in Discriminator.
    :param x: size= (batch_size, real labels)
    :param large_range: if true use the extended smoothed
    :param factor: a number in [0, 1]
    :return: If large_range = True, it returns smoothed real labels in the range [0.7, 1.2). If large_range = False,
    returns smoothed real labels in the range [1-factor, 1).
    """
    if large_range:
        return x - 0.3 + 0.5*np.random.random(size=x.shape)
    else:
        if factor < 0 or factor > 1:
            raise "Factor should be a number between 0 and 1"
        return x - factor*np.random.random(size=x.shape)


def smooth_fake_labels(x, factor=0.3):

    """
    Recommended: Apply to fake images in Discriminator.
    :param x: size= (batch_size, fake labels)
    :param factor: a number in [0, 1]
    :return: smoothed fake labels in the range (0, factor)
    """
    if factor < 0 or factor > 1:
        raise "Factor should be a number between 0 and 1."

    return x + factor*np.random.random(size=x.shape)


def create_noisy_labels(x, probability):

    """
    Recommended: Apply to real and fake images in Discriminator.
    :param x: shape=(batch_size, label)
    :param probability: probability_of_flip a number between 0 and 1
    :return: a numpy.ndarray with shape=(batch_size, 1), where about "probability" % of the elements
    are with the opposite signs.
    """
    if probability < 0 or probability > 1:
        raise "The selected number must be between 0 and 1."

    # Determine the number of labels to flip
    # ex.: if x.shape[0] = 1000, p=0.05, expected ~ 50
    n_labels = int(probability*x.shape[0])

    # Choose labels to flip
    # Create a list by list comprehension and generate a random sample from the list with size=n_labels
    flip_indices = np.random.choice([label for label in range(x.shape[0])], size=n_labels)
    # invert the labels inplace
    x[flip_indices] = 1 - x[flip_indices]
    return x


def create_labels(batch_size, large_range=False, smoothing_factor=0.1, noisy_probability=0.1):

    # Create class labels (real_image == 1, fake_image == 0), smoothed or noisy (flipped) labels
    real_labels = np.ones(shape=(batch_size, 1))
    fake_labels = np.zeros(shape=(batch_size, 1))
    smoothed_real_labels = smooth_real_labels(np.ones(shape=(batch_size, 1)), large_range=large_range, factor=smoothing_factor)
    smoothed_fake_labels = smooth_fake_labels(np.zeros(shape=(batch_size, 1)), factor=smoothing_factor)
    noisy_real_labels = create_noisy_labels(np.ones(shape=(batch_size, 1)), probability=noisy_probability)
    noisy_fake_labels = create_noisy_labels(np.zeros(shape=(batch_size, 1)), probability=noisy_probability)
    return real_labels, fake_labels, smoothed_real_labels, smoothed_fake_labels, noisy_real_labels, noisy_fake_labels


def save_my_models(discriminator, generator, gan, epoch, filepath=None):

    """
    Save the models to a specific filepath
    :param gan: GAN model
    :param generator: generator model
    :param discriminator: discriminator model
    :param filepath: specify where to save the models
    :param epoch: epoch number at which the model was saved
    :return: Save the 3 models to the given filepath
    """
    current_time = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
    discriminator.trainable = False
    gan.save(f"{filepath}_gan_{epoch}_{current_time}.h5")

    discriminator.trainable = True
    generator.save(f"{filepath}__generator_{epoch}_{current_time}.h5")
    discriminator.save(f'{filepath}_discriminator_{epoch}_{current_time}.h5')


def load_my_models(filepath=None):

    """
    :param filepath: location of the models
    :return: Three models - discriminator, generator and GAN
    """
    discriminator = tf.keras.models.load_model(f'{filepath}/model_discriminator_02-22-2023-23-38-24.h5')
    generator = tf.keras.models.load_model(f'{filepath}/model__generator_02-22-2023-23-38-24.h5')
    gan = tf.keras.models.load_model(f'{filepath}/model_gan_02-22-2023-23-38-24.h5')

    return discriminator, generator, gan


def plot_graphs(epochs,
                batch_size,
                d_loss_real,
                d_loss_fake,
                g_loss,
                d_real_acc,
                d_fake_acc,
                save_filepath="images_gan_color/graphs_c_epochs",
                current_time=time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())):

    # Plot two graphs loss/accuracy vs epochs
    my_epochs_x = range(1, epochs+1)
    plt.subplot(211)
    plt.plot(my_epochs_x, d_loss_real, color="green", label='d_loss_real_values', linewidth=0.25)
    plt.plot(my_epochs_x, d_loss_fake, color="orange", label='d_loss_fake_values', linewidth=0.25)
    plt.plot(my_epochs_x, g_loss, color='blue', label='g_loss_values', linewidth=0.25)
    plt.title("Loss values vs epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(212)
    plt.plot(my_epochs_x, d_real_acc, color='green', label='d_real_accuracy', linewidth=0.25)
    plt.plot(my_epochs_x, d_fake_acc, color='orange', label='d_fake_accuracy', linewidth=0.25)
    plt.title("Real and fake accuracies vs epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_filepath}={epochs}, batch_size={batch_size}, ct={current_time}.png")


def train(epochs=100,
          batch_size=32,
          save_interval=50,
          my_latent_dim=100,
          my_disc_optimizer=tf.keras.optimizers.Adam(),
          my_gan_optimizer=tf.keras.optimizers.Adam(),
          my_d_dropout_rate=0.25,
          my_g_dropout_rate=0.4,
          my_d_momentum=0.99,
          my_g_momentum=0.99,
          residual_models=False):

    """Takes all previous functions into one"""
    # record start time
    t_initial = time.time()

    # Define empty lists to store "loss" and "accuracy" values
    d_loss_real_list = []
    d_loss_fake_list = []
    g_loss_list = []
    d_real_accuracy_list = []
    d_fake_accuracy_list = []

    # Load the dataset
    x_train = load_my_cifar10_data()

    # Build the models
    if residual_models:
        discriminator = build_residual_discriminator(optimizer=my_disc_optimizer,
                                                     d_dropout_rate=my_d_dropout_rate,
                                                     d_momentum=my_d_momentum)
        generator = build_residual_generator(latent_dim=my_latent_dim,
                                             g_dropout_rate=my_g_dropout_rate,
                                             g_momentum=my_g_momentum)
        gcn = build_residual_gan(discriminator, generator, optimizer=my_gan_optimizer)
    else:
        discriminator = build_discriminator(optimizer=my_disc_optimizer,
                                            d_dropout_rate=my_d_dropout_rate,
                                            d_momentum=my_d_momentum)
        generator = build_generator(latent_dim=my_latent_dim,
                                    g_dropout_rate=my_g_dropout_rate,
                                    g_momentum=my_g_momentum)
        gcn = build_gan_combined_network(discriminator, generator, optimizer=my_gan_optimizer)

    # Create class labels (real_image == 1, fake_image == 0), smoothed or noisy (flipped) labels
    (real_labels, fake_labels,
     smoothed_real_labels, smoothed_fake_labels,
     noisy_real_labels, noisy_fake_labels) = create_labels(batch_size, large_range=False,
                                                           smoothing_factor=0.1,
                                                           noisy_probability=0.1)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select randomly "batch_size" number of images between indices 0 and 60000
        selected_real_images = select_real_samples(x_train, batch_size)

        # Generate a batch of new images from noise input
        generated_fake_images = generate_fake_samples(generator, batch_size, my_latent_dim)

        # Train the discriminator, first on batch with real images, then on batch with fake images
        d_loss_real = discriminator.train_on_batch(selected_real_images, smoothed_real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # averaged loss

        # d_loss_real is a list of [loss, accuracy]
        d_loss_real_list.append(d_loss_real[0])  # select loss
        d_loss_fake_list.append(d_loss_fake[0])

        d_real_accuracy_list.append(d_loss_real[1])  # select accuracy
        d_fake_accuracy_list.append(d_loss_fake[1])

        # ---------------------
        #  Train Generator
        # ---------------------

        # Generate some fake images from different noise input. Intentionally miss-label fake images with real labels.
        noise_2 = np.random.normal(loc=0, scale=1, size=(batch_size, my_latent_dim))
        g_loss = gcn.train_on_batch(noise_2, real_labels)  # use real_labels
        g_loss_list.append(g_loss)

        # Print the progress
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}] [G loss: {g_loss:.4f}]")

        # Periodically generate images and save them to the directory
        if epoch % save_interval == 0:
            plot_generated_images(epoch, generator, latent_dim=my_latent_dim,
                                  filepath="images_gan_color/gan_generated_image_epoch")

        # Save the 3 models every 10000 epochs
        if epoch % 10000 == 0:
            save_my_models(discriminator, generator, gcn, epoch, filepath="images_gan_color/model")

    # Plot two graphs loss/accuracy vs epochs
    plot_graphs(epochs, batch_size, d_loss_real_list, d_loss_fake_list, g_loss_list, d_real_accuracy_list,
                d_fake_accuracy_list)

    print(f"Elapsed time: {(time.time() - t_initial) / 60:.2f} min")
    print("Training is complete.")


def continue_training(epochs=100,
                      batch_size=32,
                      save_interval=50,
                      my_latent_dim=100,
                      my_disc_optimizer=tf.keras.optimizers.Adam(),
                      my_gan_optimizer=tf.keras.optimizers.Adam(),
                      filepath_load=None):

    """Takes all previous functions into one"""
    # record start time
    t_initial = time.time()

    # Define empty lists to store "loss" and "accuracy" values
    d_loss_real_list = []
    d_loss_fake_list = []
    g_loss_list = []
    d_real_accuracy_list = []
    d_fake_accuracy_list = []

    # Load the dataset
    x_train = load_my_cifar10_data()

    # Load the models
    discriminator, generator, gcn = load_my_models(filepath=filepath_load)

    # Create class labels (real_image == 1, fake_image == 0), smoothed or noisy (flipped) labels
    real_labels = np.ones(shape=(batch_size, 1))
    fake_labels = np.zeros(shape=(batch_size, 1))
    smoothed_real_labels = smooth_real_labels(np.ones(shape=(batch_size, 1)), large_range=False, factor=0.1)
    # smoothed_fake_labels = smooth_fake_labels(np.zeros(shape=(batch_size, 1)), factor=0.1)
    # noisy_real_labels = create_noisy_labels(np.ones(shape=(batch_size, 1)), probability_of_flip=0.1)
    # noisy_fake_labels = create_noisy_labels(np.zeros(shape=(batch_size, 1)), probability_of_flip=0.1)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select randomly "batch_size" number of images between indices 0 and 60000
        selected_real_images = select_real_samples(x_train, batch_size)

        # Generate a batch of new images from noise input
        generated_fake_images = generate_fake_samples(generator, batch_size, my_latent_dim)

        # Train the discriminator, first on batch with real images, then on batch with fake images
        d_loss_real = discriminator.train_on_batch(selected_real_images, smoothed_real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # averaged loss

        # d_loss_real is a list of [loss, accuracy]
        d_loss_real_list.append(d_loss_real[0])  # select loss
        d_loss_fake_list.append(d_loss_fake[0])

        d_real_accuracy_list.append(d_loss_real[1])  # select accuracy
        d_fake_accuracy_list.append(d_loss_fake[1])

        # ---------------------
        #  Train Generator
        # ---------------------

        # Generate some fake images from different noise input. Intentionally miss-label fake images with real labels.
        noise_2 = np.random.normal(loc=0, scale=1, size=(batch_size, my_latent_dim))
        g_loss = gcn.train_on_batch(noise_2, real_labels)  # use real_labels
        g_loss_list.append(g_loss)

        # Print the progress
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}] [G loss: {g_loss:.4f}]")

        # Periodically generate images and save them to the directory
        if epoch % save_interval == 0:
            plot_generated_images(epoch, generator,
                                  latent_dim=my_latent_dim, filepath="images_gan_color/gan_generated_image_epoch")

        # Save the 3 models every 10000 epochs
        if epoch % 10000 == 0:
            save_my_models(discriminator, generator, gcn, epoch, filepath="images_gan_color/model")

    # Plot two graphs loss/accuracy vs epochs
    plot_graphs(epochs, batch_size, d_loss_real_list, d_loss_fake_list,
                g_loss_list, d_real_accuracy_list, d_fake_accuracy_list)

    print(f"Elapsed time: {(time.time() - t_initial) / 60:.2f} min")
    print("Training is complete.")


train(epochs=100001,
      save_interval=250,
      batch_size=256,
      my_latent_dim=128,
      my_disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.5),
      my_gan_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
      my_d_dropout_rate=0.4,
      my_g_dropout_rate=0.4,
      my_d_momentum=0.99,
      my_g_momentum=0.99,
      residual_models=True)

"""continue_training(epochs=100,
                  save_interval=10,
                  batch_size=256,
                  my_latent_dim=128,
                  my_disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  my_gan_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
                  filepath_load="images_gan_color/23 Residual (no BN in D and G, d_lr=0.0002, smooth_r=0.2)")"""

# residual_models=True
# no BN in D and G
# d_lr = 0.0004, g_lr = 0.0001
# smoothed real labels