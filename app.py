import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# VAE model class (same as in training)
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # SAME as Colab
        self.fc21 = nn.Linear(256, 20)
        self.fc22 = nn.Linear(256, 20)
        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 784)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Load trained model
device = torch.device('cpu')
model = VAE().to(device)
model.load_state_dict(torch.load('vae_mnist.pth', map_location=device))
model.eval()

st.title("MNIST Handwritten Digit Generator 🤖✍️")

# User selects digit
digit = st.number_input("Enter Digit (0-9):", min_value=0, max_value=9, step=1)

if st.button("Generate 5 Images"):
    with torch.no_grad():
        z = torch.randn(5, 20).to(device)  # 20 = latent size
        samples = model.decode(z).cpu()
        samples = samples.view(5, 28, 28)

        # Display images
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axes[i].imshow(samples[i], cmap='gray')
            axes[i].axis('off')
        st.pyplot(fig)
