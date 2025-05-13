import os
import shutil
import random
from pathlib import Path

# Define constants
SOURCE_DIR = "data"
TARGET_DIR = "dataset"
TRAIN_RATIO = 0.8  # 80% for training, 20% for testing
ANIMAL_CLASSES = [
    "cane",      # dog
    "cavallo",   # horse
    "elefante",  # elephant
    "farfalla",  # butterfly
    "gallina",   # chicken
    "gatto",     # cat
    "mucca",     # cow
    "pecora",    # sheep
    "ragno",     # spider
    "scoiattolo" # squirrel
]

def create_directory_structure():
    """Create the required directory structure for the dataset."""
    # Create main directories
    os.makedirs(os.path.join(TARGET_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "test"), exist_ok=True)
    
    # Create class subdirectories
    for animal_class in ANIMAL_CLASSES:
        os.makedirs(os.path.join(TARGET_DIR, "train", animal_class), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, "test", animal_class), exist_ok=True)

def split_and_copy_images():
    """Split images into train and test sets and copy them to the target directory."""
    for animal_class in ANIMAL_CLASSES:
        source_path = os.path.join(SOURCE_DIR, animal_class)
        
        # Skip if the source directory doesn't exist
        if not os.path.exists(source_path):
            print(f"Warning: Source directory {source_path} not found. Skipping.")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(source_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle the list of images
        random.shuffle(image_files)
        
        # Calculate split point
        split_idx = int(len(image_files) * TRAIN_RATIO)
        
        # Split into training and testing sets
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]
        
        # Copy training images
        for filename in train_files:
            source_file = os.path.join(source_path, filename)
            target_file = os.path.join(TARGET_DIR, "train", animal_class, filename)
            shutil.copy2(source_file, target_file)
        
        # Copy testing images
        for filename in test_files:
            source_file = os.path.join(source_path, filename)
            target_file = os.path.join(TARGET_DIR, "test", animal_class, filename)
            shutil.copy2(source_file, target_file)
        
        print(f"Processed {animal_class}: {len(train_files)} training images, {len(test_files)} testing images")

def main():
    print("Starting dataset preparation...")
    
    # Create the directory structure
    create_directory_structure()
    
    # Split and copy images
    split_and_copy_images()
    
    print(f"\nDataset preparation complete!")
    print(f"Images organized in {TARGET_DIR}/train and {TARGET_DIR}/test")

if __name__ == "__main__":
    main() 