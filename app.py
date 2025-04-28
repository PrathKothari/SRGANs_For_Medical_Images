import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from generator import Generator

# Load the trained generator
generator = Generator()
generator.load_state_dict(torch.load("/generator_epoch_6 (2).pth", map_location='cpu'))
generator.eval()

# Define sizes
low_res_size = (128, 128)
high_res_size = (512, 512)

# Define normalizations
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
unnormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # For visualization

# Smart preprocessing function
def smart_preprocess(image):
    image = image.convert("RGB")  # Ensure RGB format

    # Resize logic
    max_size = max(image.size)
    if max_size > 512:
        scale_factor = 512 / max_size
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, Image.BICUBIC)

    min_size = min(image.size)
    if min_size < 128:
        scale_factor = 128 / min_size
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, Image.BICUBIC)

    lr_image = image.resize(low_res_size, Image.BICUBIC)

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    lr_tensor = transform(lr_image).unsqueeze(0)  # Add batch dimension
    return lr_tensor, lr_image

# De-normalization for displaying
def denormalize(tensor):
    tensor = unnormalize(tensor.squeeze(0))  # Remove batch
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

# Streamlit UI
st.title("Super-Resolution Generator ✨")
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Uploaded Image", use_container_width=True)  # fixed

    lr_tensor, lr_image = smart_preprocess(image)

    with torch.no_grad():
        sr_tensor = generator(lr_tensor)

    sr_image = denormalize(sr_tensor)

    st.image(lr_image, caption="Low-Resolution (Input to Generator)", use_container_width=True)  # fixed
    st.image(sr_image, caption="Super-Resolved Output (512x512)", use_container_width=True)  # fixed
