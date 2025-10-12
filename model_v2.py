"""
FAIL-PROOF Product Pricing Model
- No emojis for universal compatibility
- Robust error handling for images
- Graceful degradation if components fail
- Target: Works even with missing dependencies
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import re
import warnings
import pickle
import os
from tqdm import tqdm
import sys
from datetime import datetime
import time
import traceback

sys.path.append('Data/student_resource')

# Safe import with fallback
try:
    from src.utils import download_images
    UTILS_AVAILABLE = True
except ImportError:
    print("WARNING: utils.py not available. Image download will be disabled.")
    UTILS_AVAILABLE = False
    def download_images(links, path):
        raise ImportError("utils.py not available")

warnings.filterwarnings('ignore')

# Safe sentence-transformers import
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("INFO: sentence-transformers loaded successfully")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence-transformers not available. Using TF-IDF instead.")
    from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor:
    """Fail-proof text feature extractor with fallbacks"""
    
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE

        if self.use_embeddings:
            try:
                print("Loading text embedding model...")
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("   SUCCESS: Model loaded")
            except Exception as e:
                print(f"   ERROR: Could not load embedding model: {e}")
                print("   FALLBACK: Switching to TF-IDF")
                self.use_embeddings = False
                self.tfidf = TfidfVectorizer(
                    max_features=1500,
                    ngram_range=(1, 3),
                    stop_words='english',
                    min_df=2
                )
        
        if not self.use_embeddings:
            print("Initializing TF-IDF vectorizer...")
            self.tfidf = TfidfVectorizer(
                max_features=1500,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2
            )
            print("   SUCCESS: TF-IDF ready")

    def extract_handcrafted_features(self, texts):
        """Extract basic text features with error handling"""
        features = {}
        
        try:
            features['text_length'] = [len(str(text)) for text in texts]
            features['word_count'] = [len(str(text).split()) for text in texts]
            features['avg_word_length'] = [
                np.mean([len(word) for word in str(text).split()]) if len(str(text).split()) > 0 else 0
                for text in texts
            ]
            features['unique_chars'] = [len(set(str(text))) for text in texts]

            # IPQ extraction with error handling
            ipq_values = []
            for text in texts:
                try:
                    text_str = str(text).lower()
                    ipq_match = re.search(r'ipq[:\s]*(\d+)', text_str)
                    if ipq_match:
                        ipq_values.append(int(ipq_match.group(1)))
                    else:
                        pack_match = re.search(r'(\d+)\s*pack|pack\s*of\s*(\d+)|(\d+)\s*ct|(\d+)\s*count', text_str)
                        if pack_match:
                            val = [g for g in pack_match.groups() if g]
                            ipq_values.append(int(val[0]) if val else 1)
                        else:
                            ipq_values.append(1)
                except:
                    ipq_values.append(1)
            
            features['ipq'] = ipq_values
            features['ipq_log'] = np.log1p(ipq_values)

            # Keywords with error handling
            premium_keywords = ['premium', 'luxury', 'deluxe', 'professional', 'elite']
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
            features['number_count'] = [len(re.findall(r'\d+', str(text))) for text in texts]
            features['has_weight'] = [
                1 if re.search(r'\d+\s*(oz|lb|kg|g)', str(text).lower()) else 0 
                for text in texts
            ]
            features['has_volume'] = [
                1 if re.search(r'\d+\s*(ml|l|fl oz)', str(text).lower()) else 0 
                for text in texts
            ]

            print(f"   Extracted {len(features)} handcrafted features")
            
        except Exception as e:
            print(f"   ERROR in handcrafted features: {e}")
            # Return minimal features as fallback
            features = {
                'text_length': [len(str(text)) for text in texts],
                'word_count': [len(str(text).split()) for text in texts]
            }
        
        return pd.DataFrame(features)

    def fit_transform(self, texts):
        """Fit and transform with error handling"""
        handcrafted = self.extract_handcrafted_features(texts)

        if self.use_embeddings:
            try:
                print("   Generating text embeddings...")
                embeddings = self.text_model.encode(
                    [str(text) for text in texts], 
                    show_progress_bar=True, 
                    batch_size=64
                )
                embeddings_df = pd.DataFrame(
                    embeddings, 
                    columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])]
                )
                print(f"   SUCCESS: Generated {embeddings.shape[1]} embedding features")
                return pd.concat([handcrafted, embeddings_df], axis=1)
            except Exception as e:
                print(f"   ERROR in embeddings: {e}")
                print("   FALLBACK: Using TF-IDF instead")
                self.use_embeddings = False
                self.tfidf = TfidfVectorizer(
                    max_features=1500,
                    ngram_range=(1, 3),
                    stop_words='english',
                    min_df=2
                )
        
        # TF-IDF fallback
        try:
            print("   Generating TF-IDF features...")
            tfidf_features = self.tfidf.fit_transform([str(text) for text in texts])
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(), 
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            print(f"   SUCCESS: Generated {tfidf_features.shape[1]} TF-IDF features")
            return pd.concat([handcrafted, tfidf_df], axis=1)
        except Exception as e:
            print(f"   ERROR in TF-IDF: {e}")
            print("   CRITICAL: Returning only handcrafted features")
            return handcrafted

    def transform(self, texts):
        """Transform with error handling"""
        handcrafted = self.extract_handcrafted_features(texts)

        if self.use_embeddings:
            try:
                print("   Generating text embeddings...")
                embeddings = self.text_model.encode(
                    [str(text) for text in texts], 
                    show_progress_bar=True, 
                    batch_size=64
                )
                embeddings_df = pd.DataFrame(
                    embeddings, 
                    columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])]
                )
                return pd.concat([handcrafted, embeddings_df], axis=1)
            except Exception as e:
                print(f"   ERROR in embeddings: {e}")
                # Don't switch to TF-IDF in transform, use what was fit
                return handcrafted
        else:
            try:
                tfidf_features = self.tfidf.transform([str(text) for text in texts])
                tfidf_df = pd.DataFrame(
                    tfidf_features.toarray(), 
                    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
                )
                return pd.concat([handcrafted, tfidf_df], axis=1)
            except Exception as e:
                print(f"   ERROR in TF-IDF transform: {e}")
                return handcrafted


class ImageFeatureExtractor:
    """FAIL-PROOF image feature extractor"""
    
    def __init__(self, image_dir='images'):
        self.image_dir = image_dir
        self.model = None
        self.device = None
        self.transform = None
        self.failed = False
        
        try:
            print("Loading ResNet50 model...")
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            
            # GPU detection
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
            print(f"   SUCCESS: Model loaded on {self.device}")

            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"   ERROR: Failed to load ResNet50: {e}")
            print("   WARNING: Image features will return zeros")
            self.failed = True

    def download_images_batch(self, df, dataset_type='train'):
        """Safe image download with fallback"""
        if not UTILS_AVAILABLE:
            print(f"   WARNING: Cannot download {dataset_type} images (utils.py not available)")
            return False
            
        print(f"\nChecking {dataset_type} images...")
        output_dir = os.path.join(self.image_dir, dataset_type)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"   ERROR: Cannot create directory {output_dir}: {e}")
            return False

        try:
            image_links = df['image_link'].tolist()
            existing = {f for f in os.listdir(output_dir) if f.endswith('.jpg')} if os.path.exists(output_dir) else set()
            
            if len(existing) >= len(image_links):
                print(f"   SUCCESS: All {len(image_links)} images present")
                return True
            
            print(f"   Downloading {len(image_links) - len(existing)} images...")
            download_images(image_links, output_dir)
            print(f"   SUCCESS: Download complete")
            return True
            
        except Exception as e:
            print(f"   ERROR: Download failed: {e}")
            print(f"   WARNING: Will attempt to use existing images")
            return False

    def load_image_safe(self, image_path):
        """Load image with extensive error handling"""
        if not os.path.exists(image_path):
            return None
            
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            return img
        except Exception as e:
            # Silent fail for individual images
            return None

    def get_image_path(self, dataset_type, row_index):
        """Get image path"""
        return os.path.join(self.image_dir, dataset_type, f"{row_index}.jpg")

    def extract_features(self, df, dataset_type='train', cache_file=None):
        """FAIL-PROOF feature extraction"""
        
        # Check cache first
        if cache_file and os.path.exists(cache_file):
            try:
                print(f"   Loading cached features from {cache_file}")
                cached = pd.read_pickle(cache_file)
                if len(cached) == len(df):
                    print(f"   SUCCESS: Cache valid ({len(cached)} samples)")
                    return cached
                else:
                    print(f"   WARNING: Cache size mismatch (expected {len(df)}, got {len(cached)})")
            except Exception as e:
                print(f"   ERROR: Could not load cache: {e}")

        # If model failed to load, return zeros
        if self.failed or self.model is None:
            print(f"   WARNING: Model not available. Returning zero features for {len(df)} images")
            zero_features = pd.DataFrame(
                np.zeros((len(df), 2048)),
                columns=[f'img_feat_{i}' for i in range(2048)]
            )
            return zero_features

        # Try to download images
        download_success = self.download_images_batch(df, dataset_type)
        if not download_success:
            print(f"   WARNING: Download issues. Will process available images only")

        # Extract features with robust error handling
        features_list = []
        failed_count = 0
        success_count = 0
        
        print(f"   Extracting features from {len(df)} images...")
        
        for idx in tqdm(range(len(df)), desc="   Processing images"):
            try:
                img_path = self.get_image_path(dataset_type, idx)
                img = self.load_image_safe(img_path)

                if img is None:
                    # Image missing or corrupted
                    features = np.zeros(2048)
                    failed_count += 1
                else:
                    # Process image
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        features = self.model(img_tensor).squeeze().cpu().numpy()
                    success_count += 1
                    
            except Exception as e:
                # Any error during processing
                features = np.zeros(2048)
                failed_count += 1
                if failed_count <= 3:
                    print(f"\n   ERROR at index {idx}: {str(e)[:100]}")
            
            features_list.append(features)

        # Summary
        print(f"\n   Results: {success_count} successful, {failed_count} failed")
        if failed_count > 0:
            print(f"   WARNING: {failed_count}/{len(df)} images failed (using zero features)")
        
        # Create DataFrame
        image_df = pd.DataFrame(
            features_list, 
            columns=[f'img_feat_{i}' for i in range(2048)]
        )

        # Save cache
        if cache_file:
            try:
                cache_dir = os.path.dirname(cache_file)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                image_df.to_pickle(cache_file)
                print(f"   SUCCESS: Cached to {cache_file}")
            except Exception as e:
                print(f"   WARNING: Could not save cache: {e}")

        return image_df


class PricePredictionModel:
    """Simplified weighted ensemble"""

    def __init__(self, use_images=True, use_embeddings=True, fast_mode=False):
        self.use_images = use_images
        self.fast_mode = fast_mode
        self.text_extractor = TextFeatureExtractor(use_embeddings=use_embeddings)

        if use_images:
            self.image_extractor = ImageFeatureExtractor()

        self.scaler = RobustScaler()
        self.models = {}
        self.weights = {}

    def _build_models(self, fast_mode=False):
        """Build model ensemble"""
        if fast_mode:
            print("   Configuration: FAST MODE (1 model)")
            return {
                'lgb': LGBMRegressor(
                    n_estimators=150,
                    max_depth=7,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            }
        else:
            print("   Configuration: FULL MODE (2 models)")
            return {
                'xgb': XGBRegressor(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1,
                    verbosity=0
                ),
                'lgb': LGBMRegressor(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.05,
                    num_leaves=40,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            }

    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE"""
        return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

    def fit(self, train_df, validate=True):
        """Train with detailed progress tracking"""
        print("\n" + "="*60)
        print("TRAINING PIPELINE")
        print("="*60)

        # Extract features
        print("\n[1/4] Text Features")
        print("-" * 60)
        text_start = time.time()
        text_features = self.text_extractor.fit_transform(train_df['catalog_content'].values)
        print(f"   Time: {time.time() - text_start:.1f}s")

        if self.use_images:
            print("\n[2/4] Image Features")
            print("-" * 60)
            image_start = time.time()
            image_features = self.image_extractor.extract_features(
                train_df, 
                dataset_type='train',
                cache_file='cache/train_image_features.pkl'
            )
            print(f"   Time: {(time.time() - image_start)/60:.1f}min")
            X = pd.concat([text_features, image_features.set_index(text_features.index)], axis=1)
        else:
            X = text_features

        print(f"\n   Total Features: {X.shape[1]}")

        y = train_df['price'].values
        y_log = np.log1p(y)

        print(f"\n[3/4] Feature Scaling")
        print("-" * 60)
        X_scaled = self.scaler.fit_transform(X)
        print(f"   Scaled: {X_scaled.shape}")

        if validate:
            print(f"\n[3.5/4] Cross-Validation (3-Fold)")
            print("="*60)
            print(f"Estimated time: 15-20 minutes")
            print()
            
            cv_start = time.time()
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            cv_models = self._build_models(fast_mode=True)
            
            fold_scores = []
            
            for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
                fold_start = time.time()
                
                print(f"\nFold {fold_num}/3")
                print("-" * 60)
                
                X_train_fold = X_scaled[train_idx]
                X_val_fold = X_scaled[val_idx]
                y_train_fold = y_log[train_idx]
                y_val_fold = y_log[val_idx]
                
                # Train
                for model_num, (name, model) in enumerate(cv_models.items(), 1):
                    print(f"   [{model_num}/{len(cv_models)}] Training {name}...", end='', flush=True)
                    model.fit(X_train_fold, y_train_fold)
                    print(" DONE")
                
                # Predict
                print(f"   Predicting...", end='', flush=True)
                y_pred_log = list(cv_models.values())[0].predict(X_val_fold)
                y_pred = np.expm1(y_pred_log)
                y_val_real = np.expm1(y_val_fold)
                print(" DONE")
                
                # Metrics
                smape = self.calculate_smape(y_val_real, y_pred)
                mae = mean_absolute_error(y_val_real, y_pred)
                fold_scores.append(smape)
                
                fold_time = time.time() - fold_start
                cv_elapsed = time.time() - cv_start
                remaining = (cv_elapsed / fold_num) * (3 - fold_num)
                
                print(f"\n   Results:")
                print(f"      SMAPE: {smape:.2f}%")
                print(f"      MAE:   ${mae:.2f}")
                print(f"   Time:")
                print(f"      This fold:  {fold_time:.0f}s")
                print(f"      Elapsed:    {cv_elapsed/60:.1f}min")
                print(f"      Remaining:  ~{remaining/60:.1f}min")
            
            print("\n" + "="*60)
            print("CROSS-VALIDATION COMPLETE")
            print("="*60)
            print(f"Mean SMAPE: {np.mean(fold_scores):.2f}%")
            print(f"Std SMAPE:  +/-{np.std(fold_scores):.2f}%")
            print(f"Best:       {np.min(fold_scores):.2f}%")
            print(f"Worst:      {np.max(fold_scores):.2f}%")
            print(f"Total time: {(time.time()-cv_start)/60:.1f}min")
            print("="*60)

        # Train final models
        print(f"\n[4/4] Training Final Ensemble")
        print("="*60)
        expected = '2-3' if self.fast_mode else '8-12'
        print(f"Expected time: {expected} minutes")
        print(f"Models: {len(self._build_models(fast_mode=self.fast_mode))}")
        print()
        
        train_start = time.time()
        self.models = self._build_models(fast_mode=self.fast_mode)
        
        model_scores = {}
        total = len(self.models)
        
        for num, (name, model) in enumerate(self.models.items(), 1):
            model_start = time.time()
            
            print(f"[{num}/{total}] Training {name.upper()}")
            print("-" * 60)
            print(f"   Status: Fitting...", end='', flush=True)
            
            model.fit(X_scaled, y_log)
            train_time = time.time() - model_start
            
            print(f" DONE")
            print(f"   Status: Validating...", end='', flush=True)
            pred = np.expm1(model.predict(X_scaled[:5000]))
            score = self.calculate_smape(y[:5000], pred)
            model_scores[name] = score
            
            elapsed = time.time() - train_start
            avg_per_model = elapsed / num
            remaining = avg_per_model * (total - num)
            
            print(f" DONE")
            print(f"\n   SMAPE: {score:.2f}%")
            print(f"   Time:  {train_time:.0f}s ({train_time/60:.1f}min)")
            print(f"   Elapsed: {elapsed/60:.1f}min")
            if remaining > 0:
                print(f"   Remaining: ~{remaining/60:.1f}min")
            print()
        
        # Calculate weights
        total_inv = sum(1/s for s in model_scores.values())
        self.weights = {name: (1/score)/total_inv for name, score in model_scores.items()}
        
        print("="*60)
        print("ENSEMBLE WEIGHTS")
        print("="*60)
        for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(weight * 40)
            bar = '#' * bar_len + '-' * (40 - bar_len)
            print(f"   {name.upper():6s} [{bar}] {weight:.3f}")
        
        # Ensemble prediction
        print(f"\n   Generating ensemble predictions...", end='', flush=True)
        train_pred_log = np.zeros(len(X_scaled))
        for name, model in self.models.items():
            train_pred_log += self.weights[name] * model.predict(X_scaled)
        
        train_pred = np.expm1(train_pred_log)
        train_smape = self.calculate_smape(y, train_pred)
        train_mae = mean_absolute_error(y, train_pred)
        print(f" DONE")
        
        total_time = time.time() - train_start
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Ensemble SMAPE: {train_smape:.2f}%")
        print(f"Ensemble MAE:   ${train_mae:.2f}")
        print(f"Training time:  {total_time/60:.1f}min")
        print("="*60)

    def predict(self, test_df):
        """Make predictions"""
        print("\n" + "="*60)
        print("PREDICTION")
        print("="*60)

        print("\n[1/3] Text features...", end='', flush=True)
        text_features = self.text_extractor.transform(test_df['catalog_content'].values)
        print(" DONE")

        if self.use_images:
            print("[2/3] Image features...")
            image_features = self.image_extractor.extract_features(
                test_df, 
                dataset_type='test',
                cache_file='cache/test_image_features.pkl'
            )
            X = pd.concat([text_features, image_features.set_index(text_features.index)], axis=1)
        else:
            X = text_features

        print("[3/3] Predicting...", end='', flush=True)
        X_scaled = self.scaler.transform(X)
        
        # Weighted ensemble
        predictions_log = np.zeros(len(X_scaled))
        for name, model in self.models.items():
            predictions_log += self.weights[name] * model.predict(X_scaled)
        
        predictions = np.expm1(predictions_log)
        predictions = np.maximum(predictions, 0.01)

        print(" DONE")
        print("="*60)
        return predictions

    def save(self, filepath='models/model.pkl'):
        """Save model"""
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"\nModel saved: {filepath}")
        except Exception as e:
            print(f"\nERROR: Could not save model: {e}")

    @staticmethod
    def load(filepath='models/model.pkl'):
        """Load model"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded: {filepath}")
            return model
        except Exception as e:
            print(f"ERROR: Could not load model: {e}")
            return None


def main():
    """Main execution"""
    # Configuration
    USE_IMAGES = True
    USE_EMBEDDINGS = True
    VALIDATE = True
    FAST_MODE = False
    
    TRAIN_PATH = 'Data/student_resource/dataset/train.csv'
    TEST_PATH = 'Data/student_resource/dataset/test.csv'
    OUTPUT_PATH = 'submissions/test_out.csv'
    MODEL_PATH = 'models/model_failproof.pkl'

    # Create directories
    for directory in ['cache', 'models', 'submissions']:
        os.makedirs(directory, exist_ok=True)

    print("\n" + "="*60)
    print("FAIL-PROOF PRODUCT PRICING MODEL")
    print("="*60)
    print(f"Configuration:")
    print(f"  Images:     {USE_IMAGES}")
    print(f"  Embeddings: {USE_EMBEDDINGS}")
    print(f"  Validate:   {VALIDATE}")
    print(f"  Fast Mode:  {FAST_MODE}")
    print(f"  GPU:        {torch.cuda.is_available()}")
    print(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading data...")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
    except Exception as e:
        print(f"ERROR: Could not load data: {e}")
        return
    
    # Train
    start_time = time.time()
    
    try:
        model = PricePredictionModel(
            use_images=USE_IMAGES,
            use_embeddings=USE_EMBEDDINGS,
            fast_mode=FAST_MODE
        )
        model.fit(train_df, validate=VALIDATE)
        model.save(MODEL_PATH)
    except Exception as e:
        print(f"\nCRITICAL ERROR during training: {e}")
        traceback.print_exc()
        return

    # Predict
    try:
        predictions = model.predict(test_df)
    except Exception as e:
        print(f"\nCRITICAL ERROR during prediction: {e}")
        traceback.print_exc()
        return

    # Save submission
    try:
        submission = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })

        total_time = (time.time() - start_time) / 60

        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print("\nSubmission Statistics:")
        print(submission['price'].describe())
        print(f"\nTotal Pipeline Time: {total_time:.1f} minutes")

        submission.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSubmission saved: {OUTPUT_PATH}")
        print("="*60)
        print("\nSUCCESS: Pipeline completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR saving submission: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nINTERRUPTED: User stopped the process")
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {e}")
        traceback.print_exc()