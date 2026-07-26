import torch
import torchvision.transforms as transforms
from PIL import Image

class ImageInverter:
    """
    Обертка для инверсии реальных изображений с использованием e4e.
    """
    
    def __init__(self, e4e_net, device='cuda'):
        self.device = device
        self.net = e4e_net.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def invert_image(self, image_path: str):
        """Инвертирует картинку, возвращает латентный вектор W+."""
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device).float()
        
        _, latents = self.net(img_tensor, randomize_noise=False, return_latents=True)
        return latents