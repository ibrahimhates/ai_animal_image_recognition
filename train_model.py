import argparse
from model.train import train_model
from model.utils import save_class_mapping, save_metrics
import os

def main():
    """
    Main function to train the image classification model.
    """
    parser = argparse.ArgumentParser(description='Train an image classification model')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--model_name', type=str, default='resnet18',
                        choices=['resnet18', 'mobilenet_v2'],
                        help='Model architecture to use')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--save_path', type=str, default='model/model_weights.pth',
                        help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Sınıf sayısını kontrol et
    train_path = os.path.join(args.data_dir, 'train')
    test_path = os.path.join(args.data_dir, 'test')
    
    # Veri setindeki gerçek sınıf sayısını tespit et
    train_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    test_classes = len([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])
    
    # En yüksek sınıf sayısını kullan
    max_classes = max(train_classes, test_classes, args.num_classes)
    print(f"Tespit edilen sınıf sayıları: Train={train_classes}, Test={test_classes}")
    print(f"Kullanılacak sınıf sayısı: {max_classes}")
    
    # Train the model
    results = train_model(
        data_dir=args.data_dir,
        num_classes=max_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        save_path=args.save_path
    )
    
    # Save class mapping
    save_class_mapping(results['class_names'])
    
    # Save metrics to a JSON file for the Streamlit app to use
    save_metrics(
        accuracy=results['accuracy'],
        precision=results['precision'],
        recall=results['recall']
    )
    
    # Print final metrics
    print("\nEğitim tamamlandı!")
    print(f"Hayvan Türü Sayısı: {len(results['class_names'])}")
    print(f"Tanınan Hayvan Türleri: {', '.join(results['class_names'][:10])}... (ve diğerleri)")
    print(f"Doğruluk (Accuracy): {results['accuracy']:.4f}")
    print(f"Kesinlik (Precision): {results['precision']:.4f}")
    print(f"Duyarlılık (Recall): {results['recall']:.4f}")
    print(f"Model kaydedildi: {args.save_path}")
    print(f"Sınıf eşleştirmesi kaydedildi: model/class_mapping.json")

if __name__ == '__main__':
    main() 