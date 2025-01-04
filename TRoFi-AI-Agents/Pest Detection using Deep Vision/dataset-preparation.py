import os
import kaggle
import zipfile
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import json

class DatasetPreparation:
    def __init__(self, kaggle_dataset_names, base_dir='pest_detection_data'):
        """
        Initialize dataset preparation
        Args:
            kaggle_dataset_names: List of Kaggle dataset names (username/dataset-name)
            base_dir: Base directory for dataset storage
        """
        self.kaggle_dataset_names = kaggle_dataset_names
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        self.train_dir = self.processed_dir / 'train'
        self.val_dir = self.processed_dir / 'val'
        self.test_dir = self.processed_dir / 'test'

    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.raw_dir, self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def download_datasets(self):
        """Download datasets from Kaggle"""
        print("Downloading datasets from Kaggle...")
        for dataset_name in self.kaggle_dataset_names:
            try:
                kaggle.api.dataset_download_files(
                    dataset_name,
                    path=self.raw_dir,
                    unzip=True
                )
                print(f"Successfully downloaded: {dataset_name}")
            except Exception as e:
                print(f"Error downloading {dataset_name}: {str(e)}")

    def organize_data(self, split_ratio=(0.7, 0.15, 0.15)):
        """
        Organize data into train/val/test splits
        Args:
            split_ratio: Tuple of (train, val, test) ratios
        """
        # Create mapping of pest types
        pest_mapping = {}
        
        # Process each dataset
        for dataset_path in self.raw_dir.glob('*'):
            if dataset_path.is_dir():
                for class_path in dataset_path.glob('*'):
                    if class_path.is_dir():
                        class_name = class_path.name
                        image_files = list(class_path.glob('*.jpg')) + \
                                    list(class_path.glob('*.jpeg')) + \
                                    list(class_path.glob('*.png'))

                        # Split data
                        train_files, temp_files = train_test_split(
                            image_files, 
                            train_size=split_ratio[0],
                            random_state=42
                        )
                        
                        val_size = split_ratio[1] / (split_ratio[1] + split_ratio[2])
                        val_files, test_files = train_test_split(
                            temp_files,
                            train_size=val_size,
                            random_state=42
                        )

                        # Create class directories in splits
                        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
                            (split_dir / class_name).mkdir(exist_ok=True)

                        # Copy files
                        for file, split_dir in [
                            (train_files, self.train_dir),
                            (val_files, self.val_dir),
                            (test_files, self.test_dir)
                        ]:
                            for src_file in file:
                                dst_file = split_dir / class_name / src_file.name
                                shutil.copy2(src_file, dst_file)

                        # Update pest mapping
                        pest_mapping[class_name] = {
                            'total_images': len(image_files),
                            'train_images': len(train_files),
                            'val_images': len(val_files),
                            'test_images': len(test_files)
                        }

        # Save dataset statistics
        with open(self.processed_dir / 'dataset_stats.json', 'w') as f:
            json.dump(pest_mapping, f, indent=4)

        print("Data organization completed!")
        return pest_mapping

    def validate_splits(self):
        """Validate the dataset splits"""
        print("\nDataset Statistics:")
        print("-" * 50)
        
        for split in ['train', 'val', 'test']:
            split_dir = self.processed_dir / split
            total_images = sum(len(list(class_dir.glob('*'))) 
                             for class_dir in split_dir.glob('*'))
            num_classes = len(list(split_dir.glob('*')))
            
            print(f"{split.capitalize()} Set:")
            print(f"  - Total Images: {total_images}")
            print(f"  - Number of Classes: {num_classes}")
            print(f"  - Average Images per Class: {total_images/num_classes:.1f}")
            print()

    def cleanup(self):
        """Clean up raw data files"""
        if input("Clean up raw data files? (y/n): ").lower() == 'y':
            shutil.rmtree(self.raw_dir)
            print("Raw data files cleaned up!")

def main():
    # Define Kaggle datasets to download
    datasets = [
        'vbookshelf/rice-leaf-diseases-dataset',
        'vipoooool/new-plant-diseases-dataset'
    ]
    
    # Initialize dataset preparation
    data_prep = DatasetPreparation(datasets)
    
    # Setup directories
    data_prep.setup_directories()
    
    # Download datasets
    data_prep.download_datasets()
    
    # Organize data
    pest_mapping = data_prep.organize_data()
    
    # Validate splits
    data_prep.validate_splits()
    
    # Optional cleanup
    data_prep.cleanup()
    
    print("\nDataset preparation completed successfully!")
    print(f"Data is organized in: {data_prep.processed_dir}")

if __name__ == "__main__":
    main()
