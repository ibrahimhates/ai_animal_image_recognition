import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImagePreprocessor:
    """
    A class for preprocessing images for classification.
    """
    def __init__(self, image_size=(224, 224)):
        """
        Initialize the preprocessor with target image size.
        
        Args:
            image_size (tuple): Target size for the image (height, width)
        """
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image):
        """
        Preprocess an image for model input.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return self.transform(image)
    
    def preprocess_for_display(self, image):
        """
        Preprocess an image for display without normalization.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Resized image for display
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        display_transform = transforms.Compose([
            transforms.Resize(self.image_size),
        ])
        
        return display_transform(image)
    
    @staticmethod
    def load_image(image_path):
        """
        Load an image from file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            PIL.Image: Loaded image
        """
        return Image.open(image_path) 