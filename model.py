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
import requests
from io import BytesIO
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
import time

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
    
    def __init__(self):
        print("Loading image model (ResNet50)...")
        self.model = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def download_image(self, url, max_retries=3, timeout=10):
        """Download image from URL with retries"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    return img
            except Exception as e:
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)  # Wait before retry
        return None
    
    def extract_features(self, image_links, cache_file=None):
        """Extract features from images"""
        
        # Check if cached features exist
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached image features from {cache_file}")
            return pd.read_pickle(cache_file)
        
        features_list = []
        failed_count = 0
        
        print(f"Extracting features from {len(image_links)} images...")
        for idx, url in enumerate(tqdm(image_links)):
            try:
                # Download image
                img = self.download_image(url)
                
                if img is None:
                    # Use zero features for failed downloads
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
        
        print(f"Failed to download {failed_count}/{len(image_links)} images")
        
        # Convert to DataFrame
        image_df = pd.DataFrame(
            features_list,
            columns=[f'img_feat_{i}' for i in range(2048)]
        )
        
        # Cache if requested
        if cache_file:
            print(f"Caching image features to {cache_file}")
            image_df.to_pickle(cache_file)
        
        return image_df


class PricePredictionModel:
    """Main model for price prediction"""
    
    def __init__(self, use_images=True, use_embeddings=True):
        self.use_images = use_images
        self.text_extractor = TextFeatureExtractor(use_embeddings=use_embeddings)
        
        if use_images:
            self.image_extractor = ImageFeatureExtractor()
        
        self.scaler = RobustScaler()  # Better for outliers
        self.models = None
    
    def _build_models(self):
        """Build ensemble of models"""
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
            cv=5,
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
                train_df['image_link'].values,
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
            self.models = self._build_models()
            cv_scores = cross_val_score(
                self.models, X_scaled, y_log,
                cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
            )
            print(f"CV MAE (log scale): {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train final model
        print("\n[4/4] Training final model...")
        self.models = self._build_models()
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
                test_df['image_link'].values,
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
    
    print("\n" + "="*60)
    print("SMART PRODUCT PRICING CHALLENGE - ML PIPELINE")
    print("="*60)
    print(f"Use Images: {USE_IMAGES}")
    print(f"Use Embeddings: {USE_EMBEDDINGS}")
    print(f"Validation: {VALIDATE}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Price range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")
    print(f"Price median: ${train_df['price'].median():.2f}")
    
    # Initialize model
    model = PricePredictionModel(
        use_images=USE_IMAGES,
        use_embeddings=USE_EMBEDDINGS
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
    submission.to_csv('test_out.csv', index=False)
    print(f"\nSubmission saved to: test_out.csv")
    print("="*60)


if __name__ == "__main__":
    main()