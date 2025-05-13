import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def save_class_mapping(class_names, file_path='model/class_mapping.json'):
    """
    Save the mapping of class indices to class names.
    
    Args:
        class_names (list): List of class names
        file_path (str): Path to save the mapping
    """
    class_mapping = {i: name for i, name in enumerate(class_names)}
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(class_mapping, f)
    
    print(f"Class mapping saved to {file_path}")

def save_metrics(accuracy, precision, recall, file_path='model/metrics.json'):
    """
    Save model evaluation metrics to JSON file.
    
    Args:
        accuracy (float): Model accuracy
        precision (float): Model precision
        recall (float): Model recall
        file_path (str): Path to save the metrics
    """
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Metrics saved to {file_path}")

def load_class_mapping(file_path='model/class_mapping.json'):
    """
    Load the mapping of class indices to class names.
    
    Args:
        file_path (str): Path to the mapping file
    
    Returns:
        dict: Mapping of class indices to class names
    """
    with open(file_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Convert string keys to integers
    class_mapping = {int(k): v for k, v in class_mapping.items()}
    
    return class_mapping

def visualize_predictions(image, probs, class_mapping, top_k=5):
    """
    Visualize an image and its top predictions.
    
    Args:
        image (PIL.Image): Input image
        probs (numpy.ndarray): Class probabilities
        class_mapping (dict): Mapping of class indices to class names
        top_k (int): Number of top predictions to show
    
    Returns:
        matplotlib.figure.Figure: Figure with the visualization
    """
    # Get top k predictions
    top_probs, top_classes = torch.tensor(probs).topk(min(top_k, len(probs)))
    top_probs = top_probs.numpy()
    top_classes = top_classes.numpy()
    
    # Convert class indices to class names
    top_class_names = [class_mapping[idx] for idx in top_classes]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(np.array(image))
    ax1.set_title(f"Tahmin: {class_mapping[top_classes[0]]}", fontsize=14)
    ax1.axis('off')
    
    # Display bar chart of top predictions
    y_pos = np.arange(len(top_class_names))
    
    # Create horizontal bar chart with improved colors
    bars = ax2.barh(y_pos, top_probs * 100, align='center', 
                    color=plt.cm.Oranges(np.linspace(0.6, 0.8, len(top_class_names))))
    
    # Add percentage values to the end of each bar
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{prob*100:.1f}%', va='center', fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_class_names, fontsize=12)
    ax2.set_title('En Olası Meyve Türleri', fontsize=14)
    ax2.set_xlabel('Olasılık (%)', fontsize=12)
    ax2.set_xlim(0, 105)  # Leave room for percentage labels
    
    plt.tight_layout()
    
    return fig 