# Import packages
import numpy as np
import random
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import time
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

####################### Task 1: Data and Setup #######################

####### Loading data #######
# Set up training data
transform = transforms.Compose([
    transforms.ToTensor(),
])
# Set up training data
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
    target_transform=None
)

# Set up testing data
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

####### Data Information #######

# Selected device:
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected device: {device}")

# Dataset size
print(f"Training data size:{len(train_data)}")
print(f"Testing data size:{len(test_data)}")

# Image shape
image, label = train_data[0]
print(f"data image shape: {image.shape}")

# Number of classes
class_names = train_data.classes
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")


####################### Task 2: Autoencoder (AE) Implementation #######################
# 1. Create DataLoaders
# DataLoader splits the dataset into smaller batches.
# Instead of training on all 60,000 images at once,
# we train on small groups of 128 images.
BATCH_SIZE = 128

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 2. Build the Autoencoder model
# Autoencoder has two parts:
# 1. Encoder: compresses the image into a small latent representation.
# 2. Decoder: reconstructs the image from this latent representation.

# MNIST images have shape [1, 28, 28].
# Before using Linear layers, we flatten them
# 1 * 28 * 28 = 784 input features.
#
# The latent dimension is 4

class AE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 784 -> 256 -> 128 -> 64 -> 32 -> 4
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        # Decoder: 4 -> 32 -> 64 -> 128 -> 256 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),

            # Sigmoid keeps output pixel values between 0 and 1.
            # This matches MNIST images loaded with ToTensor().
            nn.Sigmoid()
        )

    def forward(self, x):
        # first compress the image to latent space
        encoded = self.encoder(x)

        # then reconstruct it 
        decoded = self.decoder(encoded)

        return decoded
    
# Create model and movee it to GPU/CPU
model_1 = AE().to(device)

# 3. Define loss function and optimizer
# We want the reconstructed image to be as close to the original as possible.
# MSELoss compares the reconstructed image with the original image.
# Lower MSE means better reconstruction.
loss_fn = nn.MSELoss()

# Adam optimizer is a popular choice for training neural networks.
# lr=0.001 is safer than 0.1 
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001, weight_decay=1e-8)

# 4. Train the Autoencoder
torch.manual_seed(42)

epochs = 10
train_losses = []

start_time = time.time()

for epoch in range(epochs):
    model_1.train()
    total_loss = 0

    for images, labels in tqdm(train_loader):
        # Move data to device
        images = images.to(device)

        # Flatten images from [batch_size, 1, 28, 28] to [batch_size, 784]
        images_flat = images.view(images.size(0), -1)

         # Forward pass:
        # input image -> encoder -> latent vector -> decoder -> reconstructed image
        reconstructed = model_1(images_flat)

        # Reconstruction loss:
        # compare reconstructed image with the original input image
        loss = loss_fn(reconstructed, images_flat)

        # Reset gradients from previous step
        optimizer.zero_grad()

        # Compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Save batch loss
        total_loss += loss.item()

    # Average loss for this epoch
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}] | Training Loss: {avg_loss:.6f}")

end_time = time.time()
training_time = end_time - start_time

# Compute test loss
model_1.eval()

test_loss = 0

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        images_flat = images.view(images.size(0), -1)

        reconstructed = model_1(images_flat)

        loss = loss_fn(reconstructed, images_flat)
        test_loss += loss.item()

test_loss = test_loss / len(test_loader)

print(f"Total training time: {training_time:.2f} seconds")
print(f"Final Train Loss: {avg_loss:.5f} | Final Test Loss: {test_loss:.5f}")

# 5. Plot training loss over epochs

plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("AE Training Loss over Epochs")
plt.grid(True)
plt.savefig("ae_training_loss.png")
plt.show()

# 6. Show reconstructed images from the test set

model_1.eval()

with torch.inference_mode():
    # Take one batch from the test set
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)

    # Flatten test images
    test_images_flat = test_images.view(test_images.size(0), -1)

    # Reconstruct test images
    reconstructed = model_1(test_images_flat)

    # Reshape reconstructed images back to image format
    reconstructed = reconstructed.view(-1, 1, 28, 28)

# Move images back to CPU for plotting
test_images = test_images.cpu()
reconstructed = reconstructed.cpu()

# Plot original and reconstructed images
n = 8

plt.figure(figsize=(12, 4))

for i in range(n):
    # Original image
    plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed image
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap="gray")
    plt.title("Recon.")
    plt.axis("off")

plt.tight_layout()
plt.savefig("ae_reconstructions.png")
plt.show()

####################### Task 3: Variational Autoencoder (VAE) #######################
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (same start as AE)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Instead of ONE latent vector we output TWO:
        # μ (mean) and log σ² (log variance)
        self.fc_mu = nn.Linear(128, 4)
        self.fc_logvar = nn.Linear(128, 4)

        # Decoder (same idea as AE)
        self.decoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

# Reparameterization trick: sample z from N(μ, σ²) using ε ~ N(0, 1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # σ = exp(log σ² / 2)
        eps = torch.randn_like(std)      # ε ~ N(0, 1)
        z = mu + eps * std              # z = μ + ε * σ
        return z
    
    def forward(self, x):
        # Encode input to get μ and log σ²
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Sample latent vector z using reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode z to reconstruct the image
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar
    
    def vae_loss(reconstructed, original, mu, logvar):
        # Reconstruction loss (MSE) (same as AE)
        recon_loss = F.mse_loss(reconstructed, original, reduction="sum")

        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total VAE loss
        return recon_loss + kl_loss

vae_model = VAE().to(device)
optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001, weight_decay=1e-8)

torch.manual_seed(42)
epochs = 10
vae_losses = []

for epoch in range(epochs):
    vae_model.train()
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        images_flat = images.view(images.size(0), -1)

        reconstructed, mu, logvar = vae_model(images_flat)

        loss = VAE.vae_loss(reconstructed, images_flat, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    vae_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}] | VAE Loss: {avg_loss:.6f}")


# VAE: Reconstructed images
vae_model.eval()

with torch.inference_mode():
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)

    test_flat = test_images.view(test_images.size(0), -1)

    reconstructed, _, _ = vae_model(test_flat)

    reconstructed = reconstructed.view(-1, 1, 28, 28)

# Move to CPU
test_images = test_images.cpu()
reconstructed = reconstructed.cpu()

# Plot
n = 8
plt.figure(figsize=(12, 4))

for i in range(n):
    # Original
    plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap="gray")
    plt.title("Recon.")
    plt.axis("off")

plt.tight_layout()
plt.savefig("vae_reconstructions.png")
plt.show()

# VAE: Generated images

# Sample random latent vectors from standard normal distribution
z = torch.randn(8, 4).to(device)  # 4 = latent dim

vae_model.eval()

with torch.inference_mode():
    generated = vae_model.decoder(z)
    generated = generated.view(-1, 1, 28, 28)

generated = generated.cpu()

# Plot generated images
plt.figure(figsize=(8, 2))

for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(generated[i].squeeze(), cmap="gray")
    plt.title("Gen")
    plt.axis("off")

plt.tight_layout()
plt.savefig("vae_generated.png")
plt.show()


####################### Task 4: Comparison and Analysis #######################

# 4.3 Latent space comparison
# Extract latent vectors from AE and VAE

model_1.eval()
vae_model.eval()

ae_latents = []
vae_latents = []
all_labels = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        # Flatten images: [batch_size, 1, 28, 28] -> [batch_size, 784]
        images_flat = images.view(images.size(0), -1)

        # AE latent representation
        ae_z = model_1.encoder(images_flat)

        # VAE latent representation
        # For VAE, we use mu as the stable latent representation
        encoded = vae_model.encoder(images_flat)
        vae_mu = vae_model.fc_mu(encoded)

        ae_latents.append(ae_z.cpu())
        vae_latents.append(vae_mu.cpu())
        all_labels.append(labels)

# Combine all batches into one tensor
ae_latents = torch.cat(ae_latents).numpy()
vae_latents = torch.cat(vae_latents).numpy()
all_labels = torch.cat(all_labels).numpy()


# Use only a subset to make t-SNE faster
n_samples = 2000

ae_latents_subset = ae_latents[:n_samples]
vae_latents_subset = vae_latents[:n_samples]
labels_subset = all_labels[:n_samples]

# Apply t-SNE to reduce latent space to 2D

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    learning_rate="auto",
    init="pca"
)

ae_latents_2d = tsne.fit_transform(ae_latents_subset)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    learning_rate="auto",
    init="pca"
)

vae_latents_2d = tsne.fit_transform(vae_latents_subset)

# Plot AE latent space

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    ae_latents_2d[:, 0],
    ae_latents_2d[:, 1],
    c=labels_subset,
    cmap="tab10",
    s=8
)
plt.colorbar(scatter)
plt.title("AE Latent Space projected with t-SNE")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.tight_layout()
plt.savefig("ae_latent_tsne.png")
plt.show()


# Plot VAE latent space

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    vae_latents_2d[:, 0],
    vae_latents_2d[:, 1],
    c=labels_subset,
    cmap="tab10",
    s=8
)
plt.colorbar(scatter)
plt.title("VAE Latent Space projected with t-SNE")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.tight_layout()
plt.savefig("vae_latent_tsne.png")
plt.show()









































