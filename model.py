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
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# No longer need to modify sys.path for utils.py

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
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2') # Smaller, faster
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
        self.model = nn.Sequential(*list(self.model.children())[:-1]) # Remove final classification layer
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

    def _download_single_image(self, args):
        """Helper function to download one image with retries."""
        url, filepath = args
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, stream=True, timeout=15)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    return True # Success
                # Don't retry on client errors like 404 Not Found
                elif 400 <= response.status_code < 500:
                    break
            except requests.exceptions.RequestException:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
        return False # Failure

    def download_images_batch(self, df, dataset_type='train'):
        """Download images in parallel using a predictable naming scheme."""
        print(f"\nDownloading {dataset_type} images...")
        output_dir = os.path.join(self.image_dir, dataset_type)
        os.makedirs(output_dir, exist_ok=True)

        tasks = []
        for _, row in df.iterrows():
            url = row['image_link']
            sample_id = row['sample_id']
            # Save all images as .jpg for consistency. PIL can handle format differences.
            filepath = os.path.join(output_dir, f"{sample_id}.jpg")
            if not os.path.exists(filepath):
                 tasks.append((url, filepath))

        if not tasks:
            print("✅ All required images are already downloaded.")
            return

        print(f"Found {len(tasks)} new images to download to: {output_dir}")
        successful_downloads = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(self._download_single_image, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Downloading images"):
                if future.result():
                    successful_downloads += 1

        print(f"✅ Downloaded {successful_downloads}/{len(tasks)} new images.")
        if successful_downloads < len(tasks):
            print(f"⚠️ Failed to download {len(tasks) - successful_downloads} images due to errors.")

    def load_image_from_disk(self, image_path):
        """Load image from disk"""
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path).convert('RGB')
                return img
            else:
                return None
        except Exception as e:
            # This can happen for corrupted files
            # print(f"Warning: Could not load image {image_path}: {e}")
            return None

    def extract_features(self, df, dataset_type='train', cache_file=None):
        """Extract features from images"""

        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached image features from {cache_file}")
            return pd.read_pickle(cache_file)

        # Download images first
        self.download_images_batch(df, dataset_type)

        features_list = []
        failed_count = 0
        image_dir = os.path.join(self.image_dir, dataset_type)
        print(f"\nExtracting features from {len(df)} images...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            img = None
            try:
                # Use the new predictable naming scheme
                sample_id = row['sample_id']
                image_path = os.path.join(image_dir, f"{sample_id}.jpg")
                img = self.load_image_from_disk(image_path)

                if img is None:
                    features = np.zeros(2048)
                    failed_count += 1
                else:
                    img_tensor = self.transform(img).unsqueeze(0)
                    with torch.no_grad():
                        features = self.model(img_tensor).squeeze().numpy()

                features_list.append(features)

            except Exception as e:
                features_list.append(np.zeros(2048)) # Fallback to zero vector on error
                failed_count += 1
                if failed_count <= 5:
                    print(f"\nError processing image for sample {row.get('sample_id', 'N/A')}: {e}")

        print(f"\n{'⚠️' if failed_count > 0 else '✅'} Failed to extract features from {failed_count}/{len(df)} images")
        if failed_count > 0:
            print(f"Note: {failed_count} images will use zero features (this may reduce accuracy)")

        # Convert to DataFrame
        image_df = pd.DataFrame(
            features_list,
            columns=[f'img_feat_{i}' for i in range(2048)]
        )

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

        self.scaler = RobustScaler() # Better for outliers
        self.models = None

    def _build_models(self, fast_mode=False):
        """Build ensemble of models"""
        if fast_mode:
            # Faster configuration for quick training
            print("Using FAST MODE model configuration...")
            base_models = [
                ('lgb', LGBMRegressor(
                    n_estimators=150,
                    max_depth=7,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ))
            ]
            meta_model = Ridge(alpha=1.0)
            stacking_cv = 2
        else:
            # Full configuration for best performance
            print("Using FULL PERFORMANCE model configuration...")
            base_models = [
                ('xgb', XGBRegressor(
                    n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42,
                    tree_method='hist', n_jobs=-1
                )),
                ('lgb', LGBMRegressor(
                    n_estimators=300, max_depth=7, learning_rate=0.05, num_leaves=40, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42,
                    n_jobs=-1, verbose=-1
                )),
                ('cat', CatBoostRegressor(
                    iterations=300, depth=7, learning_rate=0.05,
                    random_seed=42, verbose=False
                ))
            ]
            meta_model = Ridge(alpha=1.0)
            stacking_cv = 3

        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=stacking_cv,
            n_jobs=-1
        )

        return stacking

    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE metric"""
        return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

    def fit(self, train_df, validate=True):
        """Train the model"""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING PIPELINE")
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
            X = pd.concat([text_features, image_features.set_index(text_features.index)], axis=1)
        else:
            X = text_features

        print(f"\nTotal features: {X.shape[1]}")

        y = train_df['price'].values
        y_log = np.log1p(y) # Log transform target to handle skewness

        print("\n[3/4] Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        if validate:
            print("\n[3.5/4] Running cross-validation...")
            # CV is intensive, so we build a temporary model for it
            cv_model = self._build_models(fast_mode=True) # Always use fast mode for CV to save time
            
            print(f"Running CV on {len(X_scaled)} samples...")
            cv_scores = cross_val_score(
                cv_model, X_scaled, y_log,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
            print(f"\n✅ CV MAE (log scale): {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        print("\n[4/4] Training final model...")
        print("This may take some time...")
        self.models = self._build_models(fast_mode=self.fast_mode)
        self.models.fit(X_scaled, y_log)

        # Calculate training SMAPE
        train_pred = np.expm1(self.models.predict(X_scaled))
        train_smape = self.calculate_smape(y, train_pred)
        print(f"\nTraining SMAPE: {train_smape:.2f}%")

        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)

    def predict(self, test_df):
        """Make predictions"""
        print("\n" + "="*60)
        print("🚀 STARTING PREDICTION PIPELINE")
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
            X = pd.concat([text_features, image_features.set_index(text_features.index)], axis=1)
        else:
            X = text_features

        print("\n[3/3] Scaling features and making predictions...")
        X_scaled = self.scaler.transform(X)

        # Predict and inverse log transform
        predictions = np.expm1(self.models.predict(X_scaled))
        predictions = np.maximum(predictions, 0.01) # Ensure positive prices

        print("\n" + "="*60)
        print("✅ PREDICTION COMPLETE!")
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

    # ===== CONFIGURATION =====
    USE_IMAGES = True       # Set to False for faster training without images
    USE_EMBEDDINGS = True   # Set to False if sentence-transformers is not available
    VALIDATE = True         # Run cross-validation before final training
    FAST_MODE = False       # Use smaller models for quick testing. Set to False for best performance.
    
    # ===== PATHS =====
    TRAIN_PATH = 'dataset/train.csv'
    TEST_PATH = 'dataset/test.csv'
    OUTPUT_PATH = 'test_out.csv'
    # =========================

    print("\n" + "="*60)
    print("SMART PRODUCT PRICING CHALLENGE - ML PIPELINE")
    print("="*60)
    print(f"Settings: Use Images={USE_IMAGES}, Use Embeddings={USE_EMBEDDINGS}, Validate={VALIDATE}, Fast Mode={FAST_MODE}")
    
    # Load data
    print("\nLoading data...")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Make sure your paths are correct.")
        print(f"Missing file: {e.filename}")
        print("Expected structure: \n- your_script.py\n- dataset/\n  - train.csv\n  - test.csv")
        return

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Initialize and train model
    model = PricePredictionModel(
        use_images=USE_IMAGES,
        use_embeddings=USE_EMBEDDINGS,
        fast_mode=FAST_MODE
    )
    model.fit(train_df, validate=VALIDATE)
    model.save('price_prediction_model.pkl')

    # Predict
    predictions = model.predict(test_df)

    # Create submission file
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })

    print("\n" + "="*60)
    print("SUBMISSION STATISTICS")
    print("="*60)
    print(submission['price'].describe())

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Submission file saved to: {OUTPUT_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()