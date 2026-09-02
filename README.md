"""
CycleGAN for Monet-Style Image Translation
-------------------------------------------
Adapted while learning from the standard CycleGAN reference approach used in
the Kaggle "I'm Something of a Painter Myself" competition (Monet <-> Photo
unpaired image translation). This implementation follows that well-known
tutorial structure (TensorFlow/Keras, U-Net-style generator with
skip connections, PatchGAN discriminator, cycle-consistency + identity loss).

Reference: Amy Jang, "Monet CycleGAN Tutorial" (Kaggle)
https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial

Transcribed and reassembled from project report pages; not yet run
end-to-end in this environment (no GPU/TensorFlow available here) -- run
locally and verify before treating results as final.
"""

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import losses, optimizers, Model

HEIGHT_RESIZE = 256
WIDTH_RESIZE = 256
CHANNELS = 3
TRANSFORMER_BLOCKS = 6  # number of residual/transformer blocks in the bottleneck
LAMBDA = 10  # cycle-consistency loss weight


def conv_initializer():
    return tf.random_normal_initializer(0.0, 0.02)


def encoder_block(x, filters, size, strides, apply_instancenorm=True, activation=None, name=None):
    initializer = conv_initializer()
    x = L.Conv2D(filters, size, strides=strides, padding='same',
                 kernel_initializer=initializer, use_bias=False, name=name)(x)
    if apply_instancenorm:
        pass  # NOTE: add tfa.layers.InstanceNormalization() here, or
              # tf.keras.layers.GroupNormalization(groups=-1) depending on TF version
    if activation is not None:
        x = activation(x)
    return x


def decoder_block(x, filters, size, strides, apply_instancenorm=True, name=None):
    initializer = conv_initializer()
    x = L.Conv2DTranspose(filters, size, strides=strides, padding='same',
                           kernel_initializer=initializer, use_bias=False, name=name)(x)
    if apply_instancenorm:
        pass  # same InstanceNormalization note as encoder_block
    x = L.ReLU()(x)
    return x


def transformer_block(x, size, strides, name=None):
    """Residual block used in the bottleneck between encoder and decoder."""
    initializer = conv_initializer()
    skip = x
    x = L.Conv2D(x.shape[-1], size, strides=strides, padding='same',
                 kernel_initializer=initializer, use_bias=False, name=name)(x)
    x = L.ReLU()(x)
    x = L.Add()([x, skip])
    return x


def generator_fn(height=HEIGHT_RESIZE, width=WIDTH_RESIZE, channels=CHANNELS,
                  transformer_blocks=TRANSFORMER_BLOCKS):
    OUTPUT_CHANNELS = 3
    inputs = L.Input(shape=[height, width, channels], name='input_image')

    enc_1 = encoder_block(inputs, 64, 7, 1, apply_instancenorm=False,
                           activation=L.ReLU(), name='block_1')
    enc_2 = encoder_block(enc_1, 128, 3, 2, apply_instancenorm=True,
                           activation=L.ReLU(), name='block_2')
    enc_3 = encoder_block(enc_2, 256, 3, 2, apply_instancenorm=True,
                           activation=L.ReLU(), name='block_3')

    x = enc_3
    for n in range(transformer_blocks):
        x = transformer_block(x, 3, 1, name=f'block_{n + 1}')

    x_skip = L.Concatenate(name='enc_dec_skip_1')([x, enc_3])
    dec_1 = decoder_block(x_skip, 128, 3, 2, apply_instancenorm=True, name='block_1')
    x_skip = L.Concatenate(name='enc_dec_skip_2')([dec_1, enc_2])

    dec_2 = decoder_block(x_skip, 64, 3, 2, apply_instancenorm=True, name='block_2')
    x_skip = L.Concatenate(name='enc_dec_skip_3')([dec_2, enc_1])

    outputs = L.Conv2D(OUTPUT_CHANNELS, 7, strides=1, padding='same',
                        kernel_initializer=conv_initializer(), use_bias=False,
                        activation='tanh', name='decoder_output_block')(x_skip)

    generator = Model(inputs, outputs)
    return generator


def discriminator_fn(height=HEIGHT_RESIZE, width=WIDTH_RESIZE, channels=CHANNELS):
    inputs = L.Input(shape=[height, width, channels], name='input_image')

    x = encoder_block(inputs, 64, 4, 2, apply_instancenorm=False,
                       activation=L.LeakyReLU(0.2), name='block_1')
    x = encoder_block(x, 128, 4, 2, apply_instancenorm=True,
                       activation=L.LeakyReLU(0.2), name='block_2')
    x = encoder_block(x, 256, 4, 2, apply_instancenorm=True,
                       activation=L.LeakyReLU(0.2), name='block_3')
    x = encoder_block(x, 512, 4, 1, apply_instancenorm=True,
                       activation=L.LeakyReLU(0.2), name='block_4')

    outputs = L.Conv2D(1, 4, strides=1, padding='valid',
                        kernel_initializer=conv_initializer())(x)

    discriminator = Model(inputs, outputs)
    return discriminator


class CycleGan(Model):
    def __init__(self, monet_generator, photo_generator,
                 monet_discriminator, photo_discriminator, lambda_cycle=LAMBDA):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

    def compile(self, m_gen_optimizer, p_gen_optimizer,
                m_disc_optimizer, p_disc_optimizer,
                gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn


def discriminator_loss(real, generated):
    real_loss = losses.BinaryCrossentropy(from_logits=True,
                                           reduction=losses.Reduction.NONE)(tf.ones_like(real), real)
    generated_loss = losses.BinaryCrossentropy(from_logits=True,
                                                reduction=losses.Reduction.NONE)(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return losses.BinaryCrossentropy(from_logits=True,
                                      reduction=losses.Reduction.NONE)(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image, lambda_cycle):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return lambda_cycle * loss1


def identity_loss(real_image, same_image, lambda_cycle):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return lambda_cycle * 0.5 * loss


def build_and_compile_gan():
    monet_generator = generator_fn()
    photo_generator = generator_fn()
    monet_discriminator = discriminator_fn()
    photo_discriminator = discriminator_fn()

    monet_generator_optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    photo_generator_optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    monet_discriminator_optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    photo_discriminator_optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    gan_model = CycleGan(monet_generator, photo_generator,
                          monet_discriminator, photo_discriminator)

    gan_model.compile(
        m_gen_optimizer=monet_generator_optimizer,
        p_gen_optimizer=photo_generator_optimizer,
        m_disc_optimizer=monet_discriminator_optimizer,
        p_disc_optimizer=photo_discriminator_optimizer,
        gen_loss_fn=generator_loss,
        disc_loss_fn=discriminator_loss,
        cycle_loss_fn=calc_cycle_loss,
        identity_loss_fn=identity_loss,
    )
    return gan_model


if __name__ == "__main__":
    gan_model = build_and_compile_gan()
    gan_model.m_gen.summary()
    gan_model.m_disc.summary()
    print("Model built and compiled. Add a tf.data pipeline over the Monet/Photo "
          "dataset and call gan_model.fit(...) to train.")
