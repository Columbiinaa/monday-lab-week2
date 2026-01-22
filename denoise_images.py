import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

dataset_choice = "fashion"      # or "mnist"
noise_dim = 100
batch_size = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("denoising_results", exist_ok=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
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


netG = Generator().to(device)
netG.load_state_dict(torch.load("generator.pth", map_location=device))
netG.eval()

print("Generator loaded successfully")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if dataset_choice == "mnist":
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
else:
    dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

real_imgs, _ = next(iter(loader))
real_imgs = real_imgs.to(device)

def add_noise(images, noise_factor=0.6):
    noise = torch.randn_like(images) * noise_factor
    noisy = images + noise
    return torch.clamp(noisy, -1, 1)

noisy_imgs = add_noise(real_imgs)

save_image(real_imgs, "denoising_results/clean_images.png", nrow=5, normalize=True)
save_image(noisy_imgs, "denoising_results/noisy_images.png", nrow=5, normalize=True)

with torch.no_grad():
    z = torch.randn(batch_size, noise_dim).to(device)
    denoised_imgs = netG(z)

save_image(denoised_imgs, "denoising_results/denoised_images.png", nrow=5, normalize=True)

print("Denoising completed!")
