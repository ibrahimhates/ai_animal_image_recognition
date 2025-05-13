import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageClassifier:
    """
    A class that provides an interface for image classification models.
    """
    def __init__(self, num_classes, model_name='resnet18', pretrained=True):
        """
        Initialize the image classifier.
        
        Args:
            num_classes (int): Number of output classes
            model_name (str): Name of the model architecture to use
            pretrained (bool): Whether to use pretrained weights
        """
        # GPU kullanılabilirliğini kontrol et ve daha detaylı bilgi yazdır
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else "Unknown"
            print(f"GPU detected: {cuda_device_name}")
            print(f"Available GPU count: {cuda_device_count}")
            self.device = torch.device("cuda")
        else:
            print("No GPU detected, using CPU instead")
            self.device = torch.device("cpu")
            
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load base model
        if model_name == 'resnet18':
            self.model = models.resnet18(weights='DEFAULT' if pretrained else None)
            # Replace the final fully connected layer
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
            # Replace the final classifier
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # Move model to appropriate device (GPU or CPU)
        self.model = self.model.to(self.device)
        print(f"Model loaded on: {self.device}")
        
    def load_state_dict(self, state_dict_path):
        """
        Load model weights from a state dictionary.
        
        Args:
            state_dict_path (str): Path to the state dictionary file
        """
        self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        self.model.eval()
        print(f"Model weights loaded from: {state_dict_path}")
        
    def predict(self, img_tensor):
        """
        Make a prediction for an input image tensor.
        
        Args:
            img_tensor (torch.Tensor): Input image tensor of shape [1, C, H, W]
            
        Returns:
            tuple: (predicted_class_idx, class_probabilities)
        """
        self.model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            
        return predicted_idx, probabilities.squeeze().cpu().numpy() 