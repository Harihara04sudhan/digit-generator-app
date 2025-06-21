import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# --- Define the Generator Architecture (MUST be identical to the training script) ---
latent_dim = 100
num_classes = 10
img_size = 28
channels = 1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size * img_size * channels),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img

# --- Load the trained model ---
@st.cache_resource
def load_model():
    model = Generator()
    # Make sure 'generator.pth' is in the same directory as this script
    model.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

generator = load_model()

# --- Streamlit App UI ---
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using a trained Conditional GAN model.")

# User input
st.subheader("Choose a digit to generate (0-9):")
selected_digit = st.selectbox("Digit", list(range(10)))

if st.button("Generate Images"):
    st.subheader(f"Generated images of digit {selected_digit}")
    
    # Generate 5 images
    num_images_to_generate = 5
    
    # Prepare inputs for the generator
    noise = torch.randn(num_images_to_generate, latent_dim)
    labels = torch.full((num_images_to_generate,), selected_digit, dtype=torch.long)
    
    with torch.no_grad():
        generated_images = generator(noise, labels)

    # Post-process and display images
    # We need to un-normalize from [-1, 1] to [0, 1] for display
    generated_images = generated_images * 0.5 + 0.5

    cols = st.columns(num_images_to_generate)
    for i, image_tensor in enumerate(generated_images):
        with cols[i]:
            st.image(image_tensor.squeeze().numpy(), caption=f"Sample {i+1}", use_container_width=True)
