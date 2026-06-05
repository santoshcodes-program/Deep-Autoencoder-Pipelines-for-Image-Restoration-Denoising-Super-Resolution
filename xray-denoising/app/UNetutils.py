from UNetModel import DenoisingAutoencoder
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import exposure

# import cv2

def remove_noise(image_path, model_weights_path):
    model = DenoisingAutoencoder()
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()

    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((512, 736)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(image_tensor)

    output_img = (output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    p2, p98 = np.percentile(output_img, (2, 98))
    output_img_contrast_stretched = exposure.rescale_intensity(output_img, in_range=(p2, p98))

    return output_img_contrast_stretched
    