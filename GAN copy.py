# GAN


import os
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    BatchNormalization,
    Activation,
    ZeroPadding2D,
    LeakyReLU,
    UpSampling2D,
    Conv2D,
)

from keras.models import Sequential, Model
from keras.optimizers import Adam

# 1. Set random seed and output folder

np.random.seed(42)

OUTPUT_DIR = "gan_output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load CIFAR-10 data
(X, y), (_, _) = keras.datasets.cifar10.load_data()

selected_class = 1
X = X[y.flatten() == selected_class]

print(f"Selected CIFAR-10 class {selected_class}:")
print(f"Training images selected: {X.shape}")

# 3. Define Parameters
image_shape = (32, 32, 3)
latent_dimensions = 100

# 4. Build the Generator
def build_generator():
    model = Sequential(name="Generator")

    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))

    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dimensions,))
    image = model(noise)

    return Model(noise, image, name="Generator_Model")

# 5. Build the Discriminator
def build_discriminator():
    model = Sequential(name="Discriminator")

    model.add(
        Conv2D(
            32,
            kernel_size=3,
            strides=2,
            input_shape=image_shape,
            padding="same"
        )
    )
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    image = Input(shape=image_shape)
    validity = model(image)

    return Model(image, validity, name="Discriminator_Model")

# 6. Display and save generated images
def save_generated_images(epoch, generator):
    rows, cols = 4, 4
    noise = np.random.normal(0, 1, (rows * cols, latent_dimensions))
    generated_images = generator.predict(noise, verbose=0)

    # Rescale images from [-1, 1] to [0,1]
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(rows, cols, figsize=(6, 6))

    count = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis("off")
            count += 1

    fig.suptitle(f"Generated CIFAR-10 Images at Epoch {epoch}", fontsize=14)

    filename = os.path.join(OUTPUT_DIR, f"generated_epoch_{epoch}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved image grid: {filename}")

# 7. Build and compile the GAN
# 7. Build and compile the GAN

# Build and compile the discriminator first
discriminator = build_discriminator()
discriminator.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    metrics=["accuracy"],
)

# Build the generator
generator = build_generator()

# Print summaries BEFORE freezing the discriminator
print("\nGenerator Summary:")
generator.summary()

print("\nDiscriminator Summary:")
discriminator.summary()

# Freeze the discriminator only when building the combined GAN
discriminator.trainable = False

z = Input(shape=(latent_dimensions,))
generated_image = generator(z)
validity = discriminator(generated_image)

combined_network = Model(z, validity, name="Combined_GAN")
combined_network.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
)

# 8. Training the network
num_epochs = 15000
batch_size = 32

X = (X.astype(np.float32) / 127.5) - 1.0
valid = np.ones((batch_size, 1)) * 0.9
fake = np.zeros((batch_size, 1))

valid += 0.05 * np.random.random(valid.shape)
fake += 0.05 * np.random.random(fake.shape)

losses = []

for epoch in range(1, num_epochs + 1):
    # Training the Discriminator
    index = np.random.randint(0, X.shape[0], batch_size)
    real_images = X[index]

    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    generated_images = generator.predict(noise, verbose=0)

    discriminator.trainable = True

    disc_loss_real = discriminator.train_on_batch(real_images, valid)
    disc_loss_fake = discriminator.train_on_batch(generated_images, fake)

    disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

    # Train the Generator
    discriminator.trainable = False

    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    gen_loss = combined_network.train_on_batch(noise, valid)

    losses.append((epoch, disc_loss[0], disc_loss[1], gen_loss))

    # Print progress every 500 epochs
    if epoch % 500 == 0 or epoch == 1:
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Discriminator Loss: {disc_loss[0]:.4f} | "
            f"Discriminator Accuracy: {disc_loss[1] * 100:.2f}% | "
            f"Generator Loss: {float(gen_loss):.4f}"
        )

    if epoch == 1:
        save_generated_images(epoch, generator)

    if epoch == num_epochs:
        save_generated_images(epoch, generator)

# 9. Plot training losses
losses = np.array(losses, dtype=object)

epochs = [row[0] for row in losses]
discriminator_losses = [row[1] for row in losses]
generator_losses = [float(row[3]) for row in losses]

plt.figure(figsize=(10, 6))
plt.plot(epochs, discriminator_losses, label="Discriminator Loss")
plt.plot(epochs, generator_losses, label="Generator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("GAN Training Loss Over Time")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "gan_training_loss.png"), bbox_inches="tight")
plt.show()
plt.close()

print("\nTraining complete.")
print(f"First epoch image saved as: {OUTPUT_DIR}/generated_epoch_1.png")
print(f"Final epoch image saved as: {OUTPUT_DIR}/generated_epoch_{num_epochs}.png")
print(f"Loss plot saved as: {OUTPUT_DIR}/gan_training_loss.png")
