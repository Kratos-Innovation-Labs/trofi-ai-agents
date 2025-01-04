import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from concurrent.futures import ThreadPoolExecutor
import random
import glob
import os
from tqdm import tqdm
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

class PestDetectionAgent:
    def __init__(self, model_path=None):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define crop types and common pests
        self.crop_pest_mapping = {
            'rice': ['Brown Planthopper', 'Stem Borer', 'Rice Leaf Folder', 'Rice Bug'],
            'wheat': ['Aphids', 'Wheat Rust', 'Armyworm', 'Hessian Fly'],
            'potato': ['Colorado Potato Beetle', 'Potato Blight', 'Wireworm', 'Potato Tuber Moth'],
            'tomato': ['Tomato Hornworm', 'Whitefly', 'Spider Mites', 'Tomato Blight']
        }
        
        # Initialize model
        self.model = self._initialize_model()
        if model_path:
            self.load_model(model_path)
            
        # Define image transformations for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Define base augmentation transformations
        self.base_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.2, 0.2)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        ])
        
        # Define pest-specific augmentations
        self.pest_augmentations = {
            'aphids': [
                self._create_cluster_effect,
                self._add_spot_noise
            ],
            'blight': [
                self._add_lesion_effect,
                self._enhance_brown_spots
            ],
            'leaf_folder': [
                self._simulate_leaf_fold,
                self._add_pattern_noise
            ],
            'beetle': [
                self._enhance_edges,
                self._add_motion_blur
            ]
        }
        
        # Define normalization transform
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def _initialize_model(self):
        # Load pre-trained ResNet50 model
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify the final layer for pest detection
        num_classes = sum(len(pests) for pests in self.crop_pest_mapping.values())
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        return model.to(self.device)

    def preprocess_image(self, image_path):
        """
        Preprocess input image for model inference
        """
        try:
            image = Image.open(image_path).convert('RGB')
            transformed_image = self.transform(image)
            return transformed_image.unsqueeze(0).to(self.device)
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")

    def detect_pests(self, image_path, crop_type, visualize=False):
        """
        Detect pests in the given image for specified crop type
        Args:
            image_path: Path to the input image
            crop_type: Type of crop (rice, wheat, potato, tomato)
            visualize: Whether to create and save visualization
        """
        if crop_type.lower() not in self.crop_pest_mapping:
            raise ValueError(f"Unsupported crop type: {crop_type}")
        
        # Read original image for visualization
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        input_tensor = self.preprocess_image(image_path)
        
        # Model inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs)
        
        # Process results
        results = self._process_detection_results(probabilities[0], crop_type)
        
        # Create visualization if requested
        if visualize:
            self.visualize_results(original_image, results, save_path=f"{image_path}_detection.jpg")
        
        return results

    def _process_detection_results(self, probabilities, crop_type):
        """
        Process model outputs and return structured results
        """
        threshold = 0.5
        pest_list = self.crop_pest_mapping[crop_type.lower()]
        
        # Get relevant probability scores for the specified crop
        start_idx = 0
        for crop, pests in self.crop_pest_mapping.items():
            if crop == crop_type.lower():
                break
            start_idx += len(pests)
        
        crop_probabilities = probabilities[start_idx:start_idx + len(pest_list)]
        
        # Create detection results
        detections = []
        for pest, prob in zip(pest_list, crop_probabilities):
            if prob > threshold:
                detections.append({
                    'pest': pest,
                    'confidence': float(prob),
                    'severity': self._calculate_severity(float(prob))
                })
        
        return {
            'crop_type': crop_type,
            'detections': detections,
            'has_infestation': len(detections) > 0
        }

    def _calculate_severity(self, confidence):
        """
        Calculate infestation severity based on confidence score
        """
        if confidence >= 0.8:
            return 'High'
        elif confidence >= 0.6:
            return 'Medium'
        else:
            return 'Low'

    def load_model(self, model_path):
        """
        Load trained model weights
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def _generate_attention_heatmap(self, image):
        """
        Generate attention heatmap using gradient-weighted class activation mapping
        Args:
            image: Input image
        Returns:
            Numpy array representing the attention heatmap
        """
        # Convert image to tensor
        img_tensor = self.transform(Image.fromarray(image)).unsqueeze(0).to(self.device)
        
        # Register hooks for gradient computation
        gradients = []
        def save_gradients(grad):
            gradients.append(grad)
        
        # Get the last convolutional layer
        target_layer = self.model.layer4[-1].conv3
        
        # Register hook
        handle = target_layer.register_backward_hook(lambda m, i, o: save_gradients(o))
        
        # Forward pass
        self.model.eval()
        output = self.model(img_tensor)
        
        # Backward pass for the maximum score
        output.max().backward()
        
        # Get gradients and activations
        gradients = gradients[0].cpu().data.numpy()
        activations = target_layer.activations.cpu().data.numpy()
        
        # Calculate weights
        weights = np.mean(gradients, axis=(2, 3))[0, :]
        
        # Generate heatmap
        heatmap = np.zeros(activations.shape[2:])
        for i, w in enumerate(weights):
            heatmap += w * activations[0, i, :, :]
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Resize heatmap to match input image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        return heatmap

    def process_batch(self, image_paths, crop_type, batch_size=4, num_workers=4):
        """
        Process multiple images in parallel
        Args:
            image_paths: List of paths to images
            crop_type: Type of crop for detection
            batch_size: Number of images to process simultaneously
            num_workers: Number of parallel workers
        Returns:
            List of detection results
        """
        results = []
        
        def process_image(image_path):
            try:
                return self.detect_pests(image_path, crop_type, visualize=True)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                return None
        
        # Process images in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create batches
            batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
            
            # Process each batch
            for batch in tqdm(batches, desc="Processing batches"):
                batch_results = list(executor.map(process_image, batch))
                results.extend([r for r in batch_results if r is not None])
        
        return results

    def save_model(self, model_path):
        """
        Save model weights
        """
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, model_path)
        
    def visualize_results(self, image, results, save_path, include_heatmap=True):
        """
        Create visualization of pest detection results with optional heatmap
        Args:
            image: Original image (RGB format)
            results: Detection results dictionary
            save_path: Path to save the visualization
            include_heatmap: Whether to include attention heatmap
        """
        # Create figure with appropriate subplots
        num_plots = 3 if include_heatmap else 2
        plt.figure(figsize=(20, 10))
        
        # Plot original image
        plt.subplot(1, num_plots, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Generate and plot attention heatmap if requested
        if include_heatmap:
            attention_map = self._generate_attention_heatmap(image)
            plt.subplot(1, num_plots, 2)
            sns.heatmap(attention_map, cmap='YlOrRd', xticklabels=False, yticklabels=False)
            plt.title('Attention Heatmap')
            subplot_idx = 3
        else:
            subplot_idx = 2
        
        # Create detection visualization
        viz_image = image.copy()
        draw = ImageDraw.Draw(Image.fromarray(viz_image))
        
        # Add detection results
        text_y = 10
        for detection in results['detections']:
            # Define color based on severity
            color = {
                'High': (255, 0, 0),    # Red
                'Medium': (255, 165, 0), # Orange
                'Low': (255, 255, 0)     # Yellow
            }[detection['severity']]
            
            # Draw text with detection information
            text = f"{detection['pest']}: {detection['confidence']:.2f} ({detection['severity']})"
            draw.text((10, text_y), text, fill=color)
            text_y += 30
        
        # Plot annotated image
        plt.subplot(1, num_plots, subplot_idx)
        plt.imshow(viz_image)
        plt.title('Detection Results')
        plt.axis('off')
        
        # Add overall title
        plt.suptitle(f'Pest Detection Results for {results["crop_type"]}', size=16)
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def augment_image(self, image_path):
        """
        Apply data augmentation to an image
        Args:
            image_path: Path to the input image
        Returns:
            Augmented tensor
        """
        image = Image.open(image_path).convert('RGB')
        return self.augmentation(image)
    
    # Custom augmentation methods
    def _create_cluster_effect(self, image):
        """Simulate pest clustering behavior"""
        img = image.copy()
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        return img.filter(ImageFilter.GaussianBlur(radius=0.5))

    def _add_spot_noise(self, image):
        """Add random dark spots"""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        for _ in range(random.randint(10, 30)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(1, 3)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='black')
        return img

    def _add_lesion_effect(self, image):
        """Simulate disease lesions"""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        for _ in range(random.randint(5, 15)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(5, 15)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=(139, 69, 19, 128))  # Semi-transparent brown
        return img

    def _enhance_brown_spots(self, image):
        """Enhance brown coloration"""
        img = image.copy()
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(1.3)

    def _simulate_leaf_fold(self, image):
        """Simulate leaf folding effect"""
        img = image.copy()
        width, height = img.size
        # Create perspective transform matrix
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([[0, 0], [width-50, 0], [50, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_np = np.array(img)
        warped = cv2.warpPerspective(img_np, matrix, (width, height))
        return Image.fromarray(warped)

    def _add_pattern_noise(self, image):
        """Add textural patterns"""
        img = image.copy()
        img_np = np.array(img)
        noise = np.random.normal(0, 25, img_np.shape)
        noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def _enhance_edges(self, image):
        """Enhance edge detection for beetle identification"""
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    def _add_motion_blur(self, image):
        """Add motion blur effect"""
        return image.filter(ImageFilter.GaussianBlur(radius=2))

    def generate_augmented_batch(self, image_path, pest_type=None, num_augmentations=5):
        """
        Generate multiple augmented versions of an image with pest-specific augmentations
        Args:
            image_path: Path to the input image
            pest_type: Type of pest for specific augmentations
            num_augmentations: Number of augmented images to generate
        Returns:
            List of augmented tensors
        """
        augmented_images = []
        image = Image.open(image_path).convert('RGB')
        
        for _ in range(num_augmentations):
            # Apply base augmentations
            aug_image = self.base_augmentation(image)
            
            # Apply pest-specific augmentations if specified
            if pest_type and pest_type in self.pest_augmentations:
                for aug_func in self.pest_augmentations[pest_type]:
                    if random.random() > 0.5:  # 50% chance to apply each augmentation
                        aug_image = aug_func(aug_image)
            
            # Apply normalization
            aug_tensor = self.normalize(aug_image)
            augmented_images.append(aug_tensor)
            
        return augmented_images


# Example usage
def main():
    # Initialize agent
    agent = PestDetectionAgent()
    
    # Example image path and crop type
    image_path = "path/to/crop_image.jpg"
    crop_type = "rice"
    
    # Generate augmented images for training
    print("\nGenerating augmented images...")
    augmented_images = agent.generate_augmented_batch(image_path, num_augmentations=3)
    print(f"Generated {len(augmented_images)} augmented images")
    
    try:
        # Detect pests
        results = agent.detect_pests(image_path, crop_type)
        
        # Print results
        print(f"\nPest Detection Results for {results['crop_type']}:")
        print("-" * 50)
        
        if results['has_infestation']:
            print("\nDetected Pests:")
            for detection in results['detections']:
                print(f"\nPest: {detection['pest']}")
                print(f"Confidence: {detection['confidence']:.2f}")
                print(f"Severity: {detection['severity']}")
        else:
            print("\nNo pest infestation detected.")
            
    except Exception as e:
        print(f"Error during pest detection: {str(e)}")


if __name__ == "__main__":
    main()