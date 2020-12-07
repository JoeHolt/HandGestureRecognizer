# handles different image processing algorithms
import torch
from torchvision import transforms


def process_ResNetV1(img):
    """
    Processes image for Resenet-25ep model    
    """
    img_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 
    # pil first
    transformed = img_transform(img).unsqueeze(0)
    # return
    return transformed
