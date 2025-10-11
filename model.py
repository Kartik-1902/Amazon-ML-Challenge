"""
Complete ML Pipeline for Smart Product Pricing Challenge
Includes: Text processing, Image processing, Model training, and Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import re
import warnings
import pickle
import os
from tqdm import tqdm
import sys

# ===== PATH TO UTILS.PY - CHANGE THIS IF YOUR STRUCTURE IS DIFFERENT =====
sys.path.append('Data/student_resource/src')  # Add the directory containing utils.py to Python path
from utils import download_images              # Import the download_images function
# ===========================================================================

warnings.filterwarnings('ignore')

# If sentence-transformers not available, install: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Using simpler text features.")
    from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor:
    """Extract features from catalog content"""
    
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_embeddings:
            print("Loading text embedding model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller, faster
        else:
            self.tfidf = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words='english'
            )
    
    def extract_handcrafted_features(self, texts):
        """Extract handcrafted text features"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = [len(str(text)) for text in texts]
        features['word_count'] = [len(str(text).split()) for text in texts]
        features['avg_word_length'] = [
            np.mean([len(word) for word in str(text).split()]) if len(str(text).split()) > 0 else 0
            for text in texts
        ]
        
        # Extract IPQ (Item Pack Quantity)
        ipq_values = []
        for text in texts:
            text_str = str(text)
            ipq_match = re.search(r'IPQ[:\s]*(\d+)', text_str, re.IGNORECASE)
            if ipq_match:
                ipq_values.append(int(ipq_match.group(1)))
            else:
                # Try to find "pack of X" or "X pack"
                pack_match = re.search(r'(\d+)\s*pack|pack\s*of\s*(\d+)', text_str, re.IGNORECASE)
                if pack_match:
                    ipq_values.append(int(pack_match.group(1) or pack_match.group(2)))
                else:
                    ipq_values.append(1)
        features['ipq'] = ipq_values
        
        # Price indicator keywords
        premium_keywords = ['premium', 'luxury', 'deluxe', 'professional', 'pro', 'elite']
        budget_keywords = ['budget', 'value', 'basic', 'economy', 'affordable']
        
        features['has_premium_words'] = [
            sum(1 for kw in premium_keywords if kw in str(text).lower())
            for text in texts
        ]
        
        features['has_budget_words'] = [
            sum(1 for kw in budget_keywords if kw in str(text).lower())
            for text in texts
        ]
        
        # Check for brand indicators (capitalized words)
        features['capital_word_count'] = [
            len([w for w in str(text).split() if w and w[0].isupper()])
            for text in texts
        ]
        
        # Number presence (specs often indicate quality)
        features['number_count'] = [
            len(re.findall(r'\d+', str(text)))
            for text in texts
        ]
        
        return pd.DataFrame(features)
    
    def fit_transform(self, texts):
        """Fit and transform texts to features"""
        # Handcrafted features
        handcrafted = self.extract_handcrafted_features(texts)
        
        if self.use_embeddings:
            # Sentence embeddings
            print("Generating text embeddings...")
            embeddings = self.text_model.encode(
                [str(text) for text in texts],
                show_progress_bar=True,
                batch_size=32
            )
            embeddings_df = pd.DataFrame(
                embeddings,
                columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])]
            )
            return pd.concat([handcrafted, embeddings_df], axis=1)
        else:
            # TF-IDF
            print("Generating TF-IDF features...")
            tfidf_features = self.tfidf.fit_transform([str(text) for text in texts])
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            return pd.concat([handcrafted, tfidf_df], axis=1)
    
    def transform(self, texts):
        """Transform texts to features"""
        handcrafted = self.extract_handcrafted_features(texts)
        
        if self.use_embeddings:
            print("Generating text embeddings...")
            embeddings = self.text_model.encode(
                [str(text) for text in texts],
                show_progress_bar=True,
                batch_size=32
            )
            embeddings_df = pd.DataFrame(
                embeddings,
                columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])]
            )
            return pd.concat([handcrafted, embeddings_df], axis=1)
        else:
            tfidf_features = self.tfidf.transform([str(text) for text in texts])
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            return pd.concat([handcrafted, tfidf_df], axis=1)


class ImageFeatureExtractor:
    """Extract features from product images using ResNet50"""
    
    def __init__(self, image_dir='images'):
        print("Loading image model (ResNet50)...")
        self.model = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        self.image_dir = image_dir
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def download_images_batch(self, df, dataset_type='train'):
        """Download images using the provided utils.py function"""
        print(f"\nDownloading {dataset_type} images using utils.py...")
        print("Note: This may take time and might need retries due to throttling")
        
        # Create output directory for this dataset type
        output_dir = os.path.join(self.image_dir, dataset_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract image links as a list (utils.py expects a list, not DataFrame)
        image_links = df['image_link'].tolist()
        print(f"Found {len(image_links)} image links to download.")
        print(f"Images will be saved to: {output_dir}")
        
        # Use the provided download_images function
        try:
            download_images(image_links, output_dir)
            print(f"✅ Images downloaded to: {output_dir}")
        except Exception as e:
            print(f"Warning: Some images may have failed to download: {e}")
            print("Continuing with available images...")
    
    def load_image_from_disk(self, image_path):
        """Load image from disk"""
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path).convert('RGB')
                return img
            else:
                return None
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            return None
    
    def extract_features(self, df, dataset_type='train', cache_file=None):
        """Extract features from images"""
        
        # Check if cached features exist
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached image features from {cache_file}")
            return pd.read_pickle(cache_file)
        
        # Download images first using utils.py
        image_dir = os.path.join(self.image_dir, dataset_type)
        if not os.path.exists(image_dir) or len(os.listdir(image_dir)) < len(df) * 0.9:  # If less than 90% downloaded
            self.download_images_batch(df, dataset_type)
        else:
            print(f"Images already exist in {image_dir}, skipping download...")
        
        features_list = []
        failed_count = 0
        
        print(f"\nExtracting features from {len(df)} images...")
        
        # Get list of all downloaded images to match with sample_ids
        downloaded_files = set(os.listdir(image_dir)) if os.path.exists(image_dir) else set()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            try:
                # The utils.py typically saves images with index-based names or hashed names
                # We need to match them based on the order or filename pattern
                
                # Try multiple naming patterns that utils.py might use
                sample_id = row['sample_id']
                img = None
                
                # Pattern 1: Direct sample_id with extensions
                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    filename = f"{sample_id}{ext}"
                    if filename in downloaded_files:
                        image_path = os.path.join(image_dir, filename)
                        img = self.load_image_from_disk(image_path)
                        if img is not None:
                            break
                
                # Pattern 2: Index-based naming (0.jpg, 1.jpg, etc.)
                if img is None:
                    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                        filename = f"{idx}{ext}"
                        if filename in downloaded_files:
                            image_path = os.path.join(image_dir, filename)
                            img = self.load_image_from_disk(image_path)
                            if img is not None:
                                break
                
                # Pattern 3: Try to find any file that matches
                if img is None and len(downloaded_files) > 0:
                    # Get the nth file (assuming sequential download)
                    sorted_files = sorted([f for f in downloaded_files if not f.startswith('.')])
                    if idx < len(sorted_files):
                        image_path = os.path.join(image_dir, sorted_files[idx])
                        img = self.load_image_from_disk(image_path)
                
                if img is None:
                    # Use zero features for missing images
                    features = np.zeros(2048)
                    failed_count += 1
                else:
                    # Preprocess
                    img_tensor = self.transform(img).unsqueeze(0)
                    
                    # Extract features
                    with torch.no_grad():
                        features = self.model(img_tensor).squeeze().numpy()
                
                features_list.append(features)
                
            except Exception as e:
                # Fallback: zero vector
                features_list.append(np.zeros(2048))
                failed_count += 1
                if failed_count <= 5:  # Only print first 5 errors
                    print(f"\nError processing image at index {idx}: {e}")
        
        print(f"\n{'⚠️' if failed_count > 0 else '✅'} Failed to extract features from {failed_count}/{len(df)} images")
        if failed_count > 0:
            print(f"Note: {failed_count} images will use zero features (this may reduce accuracy)")
        
        # Convert to DataFrame
        image_df = pd.DataFrame(
            features_list,
            columns=[f'img_feat_{i}' for i in range(2048)]
        )
        
        # Cache if requested
        if cache_file:
            print(f"Caching image features to {cache_file}")
            os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
            image_df.to_pickle(cache_file)
        
        return image_df


class PricePredictionModel:
    """Main model for price prediction"""
    
    def __init__(self, use_images=True, use_embeddings=True, fast_mode=False):
        self.use_images = use_images
        self.fast_mode = fast_mode
        self.text_extractor = TextFeatureExtractor(use_embeddings=use_embeddings)
        
        if use_images:
            self.image_extractor = ImageFeatureExtractor()
        
        self.scaler = RobustScaler()  # Better for outliers
        self.models = None
    
    def _build_models(self, fast_mode=False):
        """Build ensemble of models"""
        if fast_mode:
            # Faster configuration for quick training
            base_models = [
                ('xgb', XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )),
                ('lgb', LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ))
            ]
        else:
            # Full configuration for best performance
            base_models = [
                ('xgb', XGBRegressor(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )),
                ('lgb', LGBMRegressor(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.05,
                    num_leaves=40,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )),
                ('cat', CatBoostRegressor(
                    iterations=300,
                    depth=7,
                    learning_rate=0.05,
                    random_seed=42,
                    verbose=False
                ))
            ]
        
        # Meta-learner
        meta_model = Ridge(alpha=1.0)
        
        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3 if fast_mode else 5,  # Fewer folds in fast mode
            n_jobs=-1
        )
        
        return stacking
    
    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE metric"""
        return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
    
    def fit(self, train_df, validate=True):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING PIPELINE")
        print("="*60)
        
        # Extract text features
        print("\n[1/4] Extracting text features...")
        text_features = self.text_extractor.fit_transform(train_df['catalog_content'].values)
        
        # Extract image features
        if self.use_images:
            print("\n[2/4] Extracting image features...")
            image_features = self.image_extractor.extract_features(
                train_df,
                dataset_type='train',
                cache_file='train_image_features.pkl'
            )
            X = pd.concat([text_features, image_features], axis=1)
        else:
            X = text_features
        
        print(f"\nTotal features: {X.shape[1]}")
        
        # Prepare target
        y = train_df['price'].values
        
        # Handle outliers in target
        y_log = np.log1p(y)  # Log transform
        
        # Scale features
        print("\n[3/4] Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Validate with cross-validation
        if validate:
            print("\n[3.5/4] Running cross-validation...")
            print("This may take 10-30 minutes depending on your hardware...")
            print("Training base models on 3-5 folds...")
            
            self.models = self._build_models(fast_mode=self.fast_mode)
            
            # Use a subset for faster CV if dataset is large
            if len(X_scaled) > 50000 and self.fast_mode:
                print("Using 20% sample for faster cross-validation...")
                sample_size = int(len(X_scaled) * 0.2)
                indices = np.random.choice(len(X_scaled), sample_size, replace=False)
                X_cv = X_scaled[indices]
                y_cv = y_log[indices]
            else:
                X_cv = X_scaled
                y_cv = y_log
            
            print(f"Running CV on {len(X_cv)} samples...")
            cv_scores = cross_val_score(
                self.models, X_cv, y_cv,
                cv=3 if self.fast_mode else 5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1  # Show progress
            )
            print(f"\n✅ CV MAE (log scale): {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train final model
        print("\n[4/4] Training final model...")
        print("This will take 5-15 minutes...")
        self.models = self._build_models(fast_mode=self.fast_mode)
        self.models.fit(X_scaled, y_log)
        
        # Calculate training SMAPE
        train_pred = np.expm1(self.models.predict(X_scaled))
        train_smape = self.calculate_smape(y, train_pred)
        print(f"\nTraining SMAPE: {train_smape:.2f}%")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
    
    def predict(self, test_df):
        """Make predictions"""
        print("\n" + "="*60)
        print("PREDICTION PIPELINE")
        print("="*60)
        
        # Extract text features
        print("\n[1/3] Extracting text features...")
        text_features = self.text_extractor.transform(test_df['catalog_content'].values)
        
        # Extract image features
        if self.use_images:
            print("\n[2/3] Extracting image features...")
            image_features = self.image_extractor.extract_features(
                test_df,
                dataset_type='test',
                cache_file='test_image_features.pkl'
            )
            X = pd.concat([text_features, image_features], axis=1)
        else:
            X = text_features
        
        # Scale features
        print("\n[3/3] Making predictions...")
        X_scaled = self.scaler.transform(X)
        
        # Predict (inverse log transform)
        predictions = np.expm1(self.models.predict(X_scaled))
        
        # Ensure positive prices
        predictions = np.maximum(predictions, 0.01)
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETE!")
        print("="*60)
        
        return predictions
    
    def save(self, filepath='model.pkl'):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath='model.pkl'):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


def main():
    """Main execution function"""
    
    # Configuration
    USE_IMAGES = True  # Set to False for faster training without images
    USE_EMBEDDINGS = True  # Set to False if sentence-transformers not available
    VALIDATE = True  # Run cross-validation
    
    # ===== CHANGE THESE PATHS IF YOUR DATA IS ELSEWHERE =====
    TRAIN_PATH = 'Data/student_resource/dataset/train.csv'
    TEST_PATH = 'Data/student_resource/dataset/test.csv'
    OUTPUT_PATH = 'test_out.csv'
    # ========================================================
    
    print("\n" + "="*60)
    print("SMART PRODUCT PRICING CHALLENGE - ML PIPELINE")
    print("="*60)
    print(f"Use Images: {USE_IMAGES}")
    print(f"Use Embeddings: {USE_EMBEDDINGS}")
    print(f"Validation: {VALIDATE}")
    print(f"\nData paths:")
    print(f"  Train: {TRAIN_PATH}")
    print(f"  Test: {TEST_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Price range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")
    print(f"Price median: ${train_df['price'].median():.2f}")
    
    # Initialize model
    model = PricePredictionModel(
        use_images=USE_IMAGES,
        use_embeddings=USE_EMBEDDINGS,
        fast_mode=FAST_MODE
    )
    
    # Train
    model.fit(train_df, validate=VALIDATE)
    
    # Save model
    model.save('price_prediction_model.pkl')
    
    # Predict
    predictions = model.predict(test_df)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Sanity checks
    print("\n" + "="*60)
    print("SUBMISSION STATISTICS")
    print("="*60)
    print(f"Total predictions: {len(submission)}")
    print(f"Predicted price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"Predicted price median: ${np.median(predictions):.2f}")
    print(f"Predicted price mean: ${np.mean(predictions):.2f}")
    
    # Save submission
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission saved to: {OUTPUT_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()