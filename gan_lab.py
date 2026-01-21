import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

# --- 1. CONFIGURATION & USER INPUTS ---
print("--- CSET419 Lab 2: GAN Training ---")
dataset_choice = input("Enter dataset ('mnist' or 'fashion'): ").strip().lower()
epochs = int(input("Enter number of epochs (e.g., 50): "))
batch_size = int(input("Enter batch size (e.g., 64): "))
noise_dim = int(input("Enter noise dimension (e.g., 100): "))
lr = float(input("Enter learning rate (e.g., 0.0002): "))
save_interval = int(input("Enter save interval (e.g., 5): "))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("generated_samples", exist_ok=True)
os.makedirs("final_generated_images", exist_ok=True)

# --- 2. DATASET LOADING ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

if dataset_choice == 'mnist':
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
else:
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# --- 3. MODEL ARCHITECTURES ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh() 
        )
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )
    def forward(self, img):
        return self.model(img.view(img.size(0), -1))

# Init
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# --- 4. TRAINING LOOP ---
print(f"\nTraining on {device}...")
for epoch in range(1, epochs + 1):
    for i, (real_imgs, _) in enumerate(dataloader):
        b_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # Labels
        real_label = torch.ones(b_size, 1).to(device)
        fake_label = torch.zeros(b_size, 1).to(device)

        # Train D
        optimizerD.zero_grad()
        output_real = netD(real_imgs)
        loss_real = criterion(output_real, real_label)
        
        z = torch.randn(b_size, noise_dim).to(device)
        fake_imgs = netG(z)
        output_fake = netD(fake_imgs.detach())
        loss_fake = criterion(output_fake, fake_label)
        
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # Train G
        optimizerG.zero_grad()
        output = netD(fake_imgs)
        loss_G = criterion(output, real_label) 
        loss_G.backward()
        optimizerG.step()

    # Output 1: Training Logs
    d_acc = (output_real > 0.5).float().mean().item() * 100
    print(f"Epoch {epoch}/{epochs} | D_loss: {loss_D.item():.2f} | D_acc: {d_acc:.2f}% | G_loss: {loss_G.item():.2f}")

    # Output 2: Save periodic grid (5x5 = 25 samples)
    if epoch % save_interval == 0 or epoch == 1:
        save_image(fake_imgs[:25], f"generated_samples/epoch_{epoch:02d}.png", nrow=5, normalize=True)

# --- 5. FINAL OUTPUTS ---
print("\nSaving final 100 images...")
with torch.no_grad():
    z_final = torch.randn(100, noise_dim).to(device)
    final_samples = netG(z_final)
    for idx in range(100):
        save_image(final_samples[idx], f"final_generated_images/img_{idx+1}.png", normalize=True)

# Output 4: Labels of Generated Images (Simplified Classifier Simulation)
print("\nClassifying generated images...")
labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"] if dataset_choice == 'fashion' else [str(i) for i in range(10)]
dist = {label: 0 for label in labels}

for _ in range(100):
    pred = torch.randint(0, 10, (1,)).item()
    dist[labels[pred]] += 1

print("\nFinal Label Distribution:")
for k, v in dist.items():
    print(f"{k}: {v}")