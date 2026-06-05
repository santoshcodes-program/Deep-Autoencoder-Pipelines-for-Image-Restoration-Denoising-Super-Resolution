from UNET import UNET
import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image

# import cv2

def super_resolve_image(image_path, model_weights_path):
    model = UNET()
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToTensor(),
        v2.ToDtype(torch.float32)
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(image_tensor)

    output_image = output_tensor.squeeze().cpu().numpy()
    return output_image
    