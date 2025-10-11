"""
Complete ML Pipeline for Smart Product Pricing Challenge - BEST PRACTICE VERSION
Includes: Text processing, Image processing with checkpoints, Model training, and Prediction
Key improvements:
- Uses official utils.py for image downloading
- Checkpointing for robust feature extraction
- Better caching strategy
- Progress tracking and recovery
- Organized file structure
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
from datetime import datetime

# Import the official download utility
sys.path.append('Data/student_resource')
from src.utils import download_images

warnings.filterwarnings('ignore')

# Try to import sentence-transformers
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
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
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

        features['capital_word_count'] = [
            len([w for w in str(text).split() if w and w[0].isupper()])
            for text in texts
        ]

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
    """Extract features from product images using ResNet50 with checkpointing"""

    def __init__(self, image_dir='images'):
        print("Loading image model (ResNet50)...")
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def download_images_batch(self, df, dataset_type='train'):
        """
        Download images using official utils.py download_images function.
        Only downloads missing images.
        """
        print(f"\n📥 Checking {dataset_type} images...")
        output_dir = os.path.join(self.image_dir, dataset_type)
        os.makedirs(output_dir, exist_ok=True)

        # Get list of image links
        image_links = df['image_link'].tolist()
        
        # Check which images already exist
        existing_images = set()
        if os.path.exists(output_dir):
            existing_images = {f for f in os.listdir(output_dir) if f.endswith('.jpg')}
        
        # Filter out images that already exist
        # Note: utils.py downloads with numbered names (0.jpg, 1.jpg, etc.)
        # We need to check if we have the right number of images
        existing_count = len(existing_images)
        required_count = len(image_links)
        
        if existing_count >= required_count:
            print(f"✅ All {required_count} images already downloaded.")
            return
        
        print(f"🔄 Downloading {required_count - existing_count} new images...")
        print(f"   Destination: {output_dir}")
        
        # Use official download_images from utils.py
        # This function handles parallel downloading efficiently
        try:
            download_images(image_links, output_dir)
            print(f"✅ Download complete!")
        except Exception as e:
            print(f"⚠️ Error during download: {e}")
            print("   Continuing with available images...")

    def load_image_from_disk(self, image_path):
        """Load image from disk"""
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path).convert('RGB')
                return img
            else:
                return None
        except Exception:
            return None

    def get_image_path(self, dataset_type, row_index):
        """
        Get image path. utils.py saves images as 0.jpg, 1.jpg, 2.jpg, etc.
        based on the order in the DataFrame.
        """
        image_dir = os.path.join(self.image_dir, dataset_type)
        image_path = os.path.join(image_dir, f"{row_index}.jpg")
        return image_path

    def extract_features(self, df, dataset_type='train', cache_file=None, 
                        force_recompute=False, checkpoint_every=500):
        """
        Extract features from images with checkpointing support
        
        Args:
            df: DataFrame with image_link and sample_id columns
            dataset_type: 'train' or 'test'
            cache_file: Path to cache file
            force_recompute: If True, ignore cache and recompute
            checkpoint_every: Save checkpoint every N images
        """
        
        # Check if final cache exists
        if cache_file and os.path.exists(cache_file) and not force_recompute:
            print(f"✅ Loading cached image features from {cache_file}")
            cached_features = pd.read_pickle(cache_file)
            
            if len(cached_features) == len(df):
                print(f"   Cache is valid ({len(cached_features)} samples)")
                return cached_features
            else:
                print(f"⚠️  Cache size mismatch. Recomputing...")

        # Setup checkpoint file
        checkpoint_file = None
        if cache_file:
            checkpoint_file = cache_file.replace('.pkl', '_checkpoint.pkl')
        
        # Try to load checkpoint
        processed_indices = set()
        checkpoint_features = {}
        start_idx = 0
        
        if checkpoint_file and os.path.exists(checkpoint_file) and not force_recompute:
            print(f"📂 Found checkpoint: {checkpoint_file}")
            try:
                checkpoint_df = pd.read_pickle(checkpoint_file)
                checkpoint_features = {idx: row.values for idx, row in checkpoint_df.iterrows()}
                processed_indices = set(checkpoint_features.keys())
                start_idx = len(processed_indices)
                print(f"   ✅ Resuming from {start_idx}/{len(df)} images")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}. Starting fresh...")

        # Download images first using official utils
        self.download_images_batch(df, dataset_type)

        features_list = [None] * len(df)  # Pre-allocate
        failed_count = 0
        
        if start_idx == 0:
            print(f"\n🔄 Extracting features from {len(df)} images...")
        else:
            print(f"\n🔄 Extracting remaining {len(df) - start_idx} images...")

        processed_count = start_idx
        
        # Reset df index to ensure we can use positional indexing
        df_reset = df.reset_index(drop=True)
        
        for idx in tqdm(range(len(df_reset)), total=len(df_reset), 
                       desc="Processing images", initial=start_idx):
            
            # Use checkpoint if available
            if idx in processed_indices:
                features_list[idx] = checkpoint_features[idx]
                continue
            
            try:
                # utils.py saves images as 0.jpg, 1.jpg, 2.jpg based on row position
                image_path = self.get_image_path(dataset_type, idx)
                img = self.load_image_from_disk(image_path)

                if img is None:
                    features = np.zeros(2048)
                    failed_count += 1
                else:
                    img_tensor = self.transform(img).unsqueeze(0)
                    with torch.no_grad():
                        features = self.model(img_tensor).squeeze().numpy()

                features_list[idx] = features
                processed_count += 1
                
                # Save checkpoint periodically
                if checkpoint_file and processed_count % checkpoint_every == 0:
                    temp_df = pd.DataFrame(
                        [f for f in features_list if f is not None],
                        columns=[f'img_feat_{i}' for i in range(2048)]
                    )
                    temp_df.to_pickle(checkpoint_file)
                    print(f"\n💾 Checkpoint: {processed_count}/{len(df)} processed")

            except Exception as e:
                features_list[idx] = np.zeros(2048)
                failed_count += 1
                if failed_count <= 3:
                    print(f"\n⚠️  Error for index {idx}: {str(e)[:50]}")

        print(f"\n{'⚠️' if failed_count > 0 else '✅'} Complete. Failed: {failed_count}/{len(df)}")
        if failed_count > 0:
            print(f"   Note: {failed_count} images will use zero features (may reduce accuracy)")

        # Convert to DataFrame
        image_df = pd.DataFrame(
            features_list,
            columns=[f'img_feat_{i}' for i in range(2048)]
        )

        # Save final cache
        if cache_file:
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            print(f"💾 Saving final cache to {cache_file}")
            image_df.to_pickle(cache_file)
            
            # Remove checkpoint after successful completion
            if checkpoint_file and os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"🗑️  Checkpoint removed (processing complete)")

        return image_df


class PricePredictionModel:
    """Main model for price prediction"""

    def __init__(self, use_images=True, use_embeddings=True, fast_mode=False):
        self.use_images = use_images
        self.fast_mode = fast_mode
        self.text_extractor = TextFeatureExtractor(use_embeddings=use_embeddings)

        if use_images:
            self.image_extractor = ImageFeatureExtractor()

        self.scaler = RobustScaler()
        self.models = None

    def _build_models(self, fast_mode=False):
        """Build ensemble of models"""
        if fast_mode:
            print("Using FAST MODE configuration...")
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
            print("Using FULL PERFORMANCE configuration...")
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
                cache_file='cache/train_image_features.pkl',
                checkpoint_every=500
            )
            X = pd.concat([text_features, image_features.set_index(text_features.index)], axis=1)
        else:
            X = text_features

        print(f"\nTotal features: {X.shape[1]}")

        y = train_df['price'].values
        y_log = np.log1p(y)

        print("\n[3/4] Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        if validate:
            print("\n[3.5/4] Running cross-validation...")
            cv_model = self._build_models(fast_mode=True)
            
            cv_scores = cross_val_score(
                cv_model, X_scaled, y_log,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
            print(f"\n✅ CV MAE (log): {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        print("\n[4/4] Training final model...")
        self.models = self._build_models(fast_mode=self.fast_mode)
        self.models.fit(X_scaled, y_log)

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

        print("\n[1/3] Extracting text features...")
        text_features = self.text_extractor.transform(test_df['catalog_content'].values)

        if self.use_images:
            print("\n[2/3] Extracting image features...")
            image_features = self.image_extractor.extract_features(
                test_df,
                dataset_type='test',
                cache_file='cache/test_image_features.pkl',
                checkpoint_every=500
            )
            X = pd.concat([text_features, image_features.set_index(text_features.index)], axis=1)
        else:
            X = text_features

        print("\n[3/3] Making predictions...")
        X_scaled = self.scaler.transform(X)

        predictions = np.expm1(self.models.predict(X_scaled))
        predictions = np.maximum(predictions, 0.01)

        print("\n" + "="*60)
        print("✅ PREDICTION COMPLETE!")
        print("="*60)

        return predictions

    def save(self, filepath='models/model.pkl'):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Model saved to {filepath}")

    @staticmethod
    def load(filepath='models/model.pkl'):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"📂 Model loaded from {filepath}")
        return model


def main():
    """Main execution function"""

    # ===== CONFIGURATION =====
    USE_IMAGES = True
    USE_EMBEDDINGS = True
    VALIDATE = True
    FAST_MODE = False
    
    # ===== PATHS =====
    TRAIN_PATH = 'Data/student_resource/dataset/train.csv'
    TEST_PATH = 'Data/student_resource/dataset/test.csv'
    OUTPUT_PATH = 'submissions/test_out.csv'
    MODEL_PATH = 'models/price_prediction_model.pkl'
    # =========================

    # Create directories
    os.makedirs('cache', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)

    print("\n" + "="*60)
    print("SMART PRODUCT PRICING - ML PIPELINE (BEST PRACTICE)")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Use Images: {USE_IMAGES}")
    print(f"  - Use Embeddings: {USE_EMBEDDINGS}")
    print(f"  - Validate: {VALIDATE}")
    print(f"  - Fast Mode: {FAST_MODE}")
    print(f"  - Using official utils.py for downloads")
    print(f"  - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n📂 Loading data...")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found: {e.filename}")
        print("Expected structure:")
        print("  - Data/student_resource/dataset/train.csv")
        print("  - Data/student_resource/dataset/test.csv")
        return

    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Initialize and train model
    model = PricePredictionModel(
        use_images=USE_IMAGES,
        use_embeddings=USE_EMBEDDINGS,
        fast_mode=FAST_MODE
    )
    model.fit(train_df, validate=VALIDATE)
    model.save(MODEL_PATH)

    # Predict
    predictions = model.predict(test_df)

    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })

    print("\n" + "="*60)
    print("📊 SUBMISSION STATISTICS")
    print("="*60)
    print(submission['price'].describe())

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Submission saved to: {OUTPUT_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()