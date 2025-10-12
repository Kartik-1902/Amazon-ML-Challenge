"""
OPTIMIZED FAIL-PROOF Product Pricing Model
- Hyperparameter tuning with Optuna
- Enhanced regularization
- Stronger ensemble with CatBoost
- Additional low-cost features
- Improved efficiency and bug fixes
- Target: SMAPE < 30% in under 60 minutes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, train_test_split
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

# Optuna for hyperparameter tuning
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    print("INFO: Optuna loaded successfully")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: Optuna not available. Using default hyperparameters.")

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
    """Enhanced text feature extractor with additional features"""
    
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
        """Extract enhanced text features with error handling"""
        features = {}
        
        try:
            # Basic text statistics
            features['text_length'] = [len(str(text)) for text in texts]
            features['word_count'] = [len(str(text).split()) for text in texts]
            features['avg_word_length'] = [
                np.mean([len(word) for word in str(text).split()]) if len(str(text).split()) > 0 else 0
                for text in texts
            ]
            features['unique_chars'] = [len(set(str(text))) for text in texts]

            # NEW FEATURE 1: Punctuation count (low-cost, high signal for product descriptions)
            features['punc_count'] = [len(re.findall(r'[!?,.]', str(text))) for text in texts]
            
            # NEW FEATURE 2: Capital letter ratio (indicates brand names, emphasis)
            features['capital_ratio'] = [
                sum(1 for c in str(text) if c.isupper()) / len(str(text)) if len(str(text)) > 0 else 0 
                for text in texts
            ]
            
            # NEW FEATURE 3: Digit to text ratio (product specs often have numbers)
            features['digit_ratio'] = [
                sum(1 for c in str(text) if c.isdigit()) / len(str(text)) if len(str(text)) > 0 else 0
                for text in texts
            ]

            # IPQ extraction with enhanced error handling
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

            # Enhanced keyword features
            premium_keywords = ['premium', 'luxury', 'deluxe', 'professional', 'elite', 'supreme']
            budget_keywords = ['budget', 'value', 'basic', 'economy', 'affordable', 'cheap']
            quality_keywords = ['organic', 'natural', 'fresh', 'pure', 'authentic']
            
            features['has_premium_words'] = [
                sum(1 for kw in premium_keywords if kw in str(text).lower()) 
                for text in texts
            ]
            features['has_budget_words'] = [
                sum(1 for kw in budget_keywords if kw in str(text).lower()) 
                for text in texts
            ]
            features['has_quality_words'] = [
                sum(1 for kw in quality_keywords if kw in str(text).lower())
                for text in texts
            ]
            
            features['capital_word_count'] = [
                len([w for w in str(text).split() if w and w[0].isupper()]) 
                for text in texts
            ]
            features['number_count'] = [len(re.findall(r'\d+', str(text))) for text in texts]
            features['has_weight'] = [
                1 if re.search(r'\d+\s*(oz|lb|kg|g|gram)', str(text).lower()) else 0 
                for text in texts
            ]
            features['has_volume'] = [
                1 if re.search(r'\d+\s*(ml|l|liter|fl oz|gallon)', str(text).lower()) else 0 
                for text in texts
            ]

            print(f"   Extracted {len(features)} handcrafted features")
            
        except Exception as e:
            print(f"   ERROR in handcrafted features: {e}")
            # Minimal fallback features
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
        """
        Transform with error handling
        BUG FIX: Correctly use already-fitted TF-IDF model without re-fitting
        """
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
                # Return handcrafted only, maintaining consistency with training
                return handcrafted
        else:
            try:
                # FIXED: Use transform (not fit_transform) on already-fitted TF-IDF
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
    """FAIL-PROOF image feature extractor with optimizations"""
    
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
            
            # GPU detection with better error handling
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
            # Enable optimizations for inference
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
            
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
        except Exception:
            # Silent fail for individual images
            return None

    def get_image_path(self, dataset_type, row_index):
        """Get image path"""
        return os.path.join(self.image_dir, dataset_type, f"{row_index}.jpg")

    def extract_features(self, df, dataset_type='train', cache_file=None):
        """FAIL-PROOF feature extraction with batch processing optimization"""
        
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
    """Enhanced model with Optuna tuning and CatBoost"""

    def __init__(self, use_images=True, use_embeddings=True, fast_mode=False, tune_hyperparams=True):
        self.use_images = use_images
        self.fast_mode = fast_mode
        self.tune_hyperparams = tune_hyperparams and OPTUNA_AVAILABLE
        self.text_extractor = TextFeatureExtractor(use_embeddings=use_embeddings)

        if use_images:
            self.image_extractor = ImageFeatureExtractor()

        self.scaler = RobustScaler()  # VERIFIED: Correctly used with fit_transform/transform
        self.models = {}
        self.weights = {}
        self.best_params = {}  # Store tuned hyperparameters

    def tune_hyperparameters(self, X_sample, y_sample):
        """
        Optuna-based hyperparameter tuning on a data subset
        Runs efficiently (25-30 trials) to find optimal parameters
        """
        if not OPTUNA_AVAILABLE:
            print("   WARNING: Optuna not available. Using default parameters.")
            return {}
        
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING WITH OPTUNA")
        print("="*60)
        print(f"Tuning on {len(X_sample)} samples")
        print(f"Running 25 trials per model (estimated: 5-8 minutes)")
        print()
        
        best_params = {}
        
        # Tune LightGBM
        def lgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 400),
                'max_depth': trial.suggest_int('max_depth', 6, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08),
                'num_leaves': trial.suggest_int('num_leaves', 30, 60),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0),  # L1 regularization
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),  # L2 regularization
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = LGBMRegressor(**params)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X_sample):
                X_train, X_val = X_sample[train_idx], X_sample[val_idx]
                y_train, y_val = y_sample[train_idx], y_sample[val_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, pred)
                scores.append(mae)
            
            return np.mean(scores)
        
        print("[1/3] Tuning LightGBM...")
        lgb_study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        lgb_study.optimize(lgb_objective, n_trials=25, show_progress_bar=True)
        best_params['lgb'] = lgb_study.best_params
        print(f"   Best LGB MAE: {lgb_study.best_value:.4f}")
        
        # Tune XGBoost
        def xgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 400),
                'max_depth': trial.suggest_int('max_depth', 6, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 2.0),  # L1 regularization
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),  # L2 regularization
                'gamma': trial.suggest_float('gamma', 0.0, 0.2),
                'random_state': 42,
                'tree_method': 'hist',
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = XGBRegressor(**params)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X_sample):
                X_train, X_val = X_sample[train_idx], X_sample[val_idx]
                y_train, y_val = y_sample[train_idx], y_sample[val_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, pred)
                scores.append(mae)
            
            return np.mean(scores)
        
        print("\n[2/3] Tuning XGBoost...")
        xgb_study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        xgb_study.optimize(xgb_objective, n_trials=25, show_progress_bar=True)
        best_params['xgb'] = xgb_study.best_params
        print(f"   Best XGB MAE: {xgb_study.best_value:.4f}")
        
        # Tune CatBoost
        def cat_objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 200, 400),
                'depth': trial.suggest_int('depth', 6, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 5.0),  # L2 regularization
                'random_seed': 42,
                'verbose': False
            }
            
            # Use GPU if available
            if torch.cuda.is_available():
                params['task_type'] = 'GPU'
            
            model = CatBoostRegressor(**params)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X_sample):
                X_train, X_val = X_sample[train_idx], X_sample[val_idx]
                y_train, y_val = y_sample[train_idx], y_sample[val_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, pred)
                scores.append(mae)
            
            return np.mean(scores)
        
        print("\n[3/3] Tuning CatBoost...")
        cat_study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        cat_study.optimize(cat_objective, n_trials=25, show_progress_bar=True)
        best_params['cat'] = cat_study.best_params
        print(f"   Best CAT MAE: {cat_study.best_value:.4f}")
        
        print("\n" + "="*60)
        print("TUNING COMPLETE")
        print("="*60)
        
        return best_params

    def _build_models(self, fast_mode=False):
        """Build enhanced ensemble with CatBoost and tuned parameters"""
        if fast_mode:
            print("   Configuration: FAST MODE (1 model)")
            return {
                'lgb': LGBMRegressor(
                    n_estimators=150,
                    max_depth=7,
                    learning_rate=0.1,
                    num_leaves=31,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            }
        else:
            print("   Configuration: FULL MODE (3 models with enhanced regularization)")
            
            # Use tuned parameters if available, otherwise use strong defaults
            lgb_params = self.best_params.get('lgb', {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 50,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 1.0,  # Enhanced L1
                'reg_lambda': 2.0,  # Enhanced L2
            })
            lgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
            
            xgb_params = self.best_params.get('xgb', {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 1.0,  # Enhanced L1
                'reg_lambda': 2.0,  # Enhanced L2
                'gamma': 0.1,
            })
            xgb_params.update({'random_state': 42, 'tree_method': 'hist', 'n_jobs': -1, 'verbosity': 0})
            
            cat_params = self.best_params.get('cat', {
                'iterations': 300,
                'depth': 8,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3.0,  # Enhanced L2
            })
            cat_params.update({'random_seed': 42, 'verbose': False})
            
            # Add GPU support for CatBoost if available
            if torch.cuda.is_available():
                cat_params['task_type'] = 'GPU'
                print("   INFO: CatBoost will use GPU")
            
            return {
                'xgb': XGBRegressor(**xgb_params),
                'lgb': LGBMRegressor(**lgb_params),
                'cat': CatBoostRegressor(**cat_params)  # NEW: CatBoost added
            }

    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE"""
        return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

    def fit(self, train_df, validate=True):
        """Train with Optuna tuning and detailed progress tracking"""
        print("\n" + "="*60)
        print("OPTIMIZED TRAINING PIPELINE")
        print("="*60)

        # Extract features
        print("\n[1/5] Text Features")
        print("-" * 60)
        text_start = time.time()
        text_features = self.text_extractor.fit_transform(train_df['catalog_content'].values)
        print(f"   Time: {time.time() - text_start:.1f}s")

        if self.use_images:
            print("\n[2/5] Image Features")
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

        print(f"\n[3/5] Feature Scaling")
        print("-" * 60)
        # BUG FIX VERIFIED: Using fit_transform on training data
        X_scaled = self.scaler.fit_transform(X)
        print(f"   Scaled: {X_scaled.shape}")

        # Hyperparameter tuning on subset
        if self.tune_hyperparams and not self.fast_mode:
            print(f"\n[3.5/5] Hyperparameter Tuning")
            print("-" * 60)
            # Use first 20,000 samples for tuning
            tune_size = min(20000, len(X_scaled))
            X_tune = X_scaled[:tune_size]
            y_tune = y_log[:tune_size]
            
            self.best_params = self.tune_hyperparameters(X_tune, y_tune)

        if validate:
            print(f"\n[4/5] Cross-Validation (3-Fold)")
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
        print(f"\n[5/5] Training Final Ensemble")
        print("="*60)
        expected = '2-3' if self.fast_mode else '10-15'
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
            # Validate on first 5000 samples for speed
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
        
        # Calculate inverse-SMAPE weights (better models get higher weight)
        total_inv = sum(1/s for s in model_scores.values())
        self.weights = {name: (1/score)/total_inv for name, score in model_scores.items()}
        
        print("="*60)
        print("ENSEMBLE WEIGHTS (Performance-Based)")
        print("="*60)
        for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(weight * 40)
            bar = '#' * bar_len + '-' * (40 - bar_len)
            print(f"   {name.upper():6s} [{bar}] {weight:.3f} ({weight*100:.1f}%)")
        
        # Ensemble prediction on full training set
        print(f"\n   Generating ensemble predictions...", end='', flush=True)
        train_pred_log = np.zeros(len(X_scaled))
        for name, model in self.models.items():
            train_pred_log += self.weights[name] * model.predict(X_scaled)
        
        train_pred = np.expm1(train_pred_log)
        train_smape = self.calculate_smape(y, train_pred)
        train_mae = mean_absolute_error(y, train_pred)
        train_rmse = np.sqrt(np.mean((y - train_pred) ** 2))
        print(f" DONE")
        
        total_time = time.time() - train_start
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Final Metrics:")
        print(f"   Ensemble SMAPE: {train_smape:.2f}%")
        print(f"   Ensemble MAE:   ${train_mae:.2f}")
        print(f"   Ensemble RMSE:  ${train_rmse:.2f}")
        print(f"Training time:     {total_time/60:.1f}min")
        print("="*60)
        
        # Performance analysis
        if train_smape > 40:
            print("\nWARNING: SMAPE > 40%. Consider:")
            print("   - Check if images are working (should see 'successful' count)")
            print("   - Increase model complexity")
            print("   - Add more feature engineering")
        elif train_smape < 25:
            print("\nEXCELLENT: SMAPE < 25%. Model is performing well!")
        else:
            print("\nGOOD: SMAPE in acceptable range (25-40%)")

    def predict(self, test_df):
        """Make predictions with proper scaling"""
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
        # BUG FIX VERIFIED: Using transform (not fit_transform) on test data
        X_scaled = self.scaler.transform(X)
        
        # Weighted ensemble prediction
        predictions_log = np.zeros(len(X_scaled))
        for name, model in self.models.items():
            predictions_log += self.weights[name] * model.predict(X_scaled)
        
        predictions = np.expm1(predictions_log)
        predictions = np.maximum(predictions, 0.01)  # Ensure positive prices

        print(" DONE")
        print("="*60)
        return predictions

    def save(self, filepath='models/model.pkl'):
        """Save model with metadata"""
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            
            # Save model with metadata
            model_data = {
                'model': self,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'best_params': self.best_params,
                'weights': self.weights
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"\nModel saved: {filepath}")
            print(f"   Includes: tuned hyperparameters and ensemble weights")
        except Exception as e:
            print(f"\nERROR: Could not save model: {e}")

    @staticmethod
    def load(filepath='models/model.pkl'):
        """Load model with metadata"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                model = model_data['model']
                print(f"Model loaded: {filepath}")
                print(f"   Trained: {model_data.get('timestamp', 'Unknown')}")
                return model
            else:
                # Backward compatibility
                print(f"Model loaded: {filepath}")
                return model_data
        except Exception as e:
            print(f"ERROR: Could not load model: {e}")
            return None


def main():
    """Main execution with optimizations"""
    # Configuration
    USE_IMAGES = True
    USE_EMBEDDINGS = True
    VALIDATE = True
    FAST_MODE = False
    TUNE_HYPERPARAMS = True  # Enable Optuna tuning
    
    TRAIN_PATH = 'Data/student_resource/dataset/train.csv'
    TEST_PATH = 'Data/student_resource/dataset/test.csv'
    OUTPUT_PATH = 'submissions/test_out_optimized.csv'
    MODEL_PATH = 'models/model_optimized.pkl'

    # Create directories
    for directory in ['cache', 'models', 'submissions']:
        os.makedirs(directory, exist_ok=True)

    print("\n" + "="*60)
    print("OPTIMIZED FAIL-PROOF PRODUCT PRICING MODEL")
    print("="*60)
    print(f"Configuration:")
    print(f"  Images:         {USE_IMAGES}")
    print(f"  Embeddings:     {USE_EMBEDDINGS}")
    print(f"  Validate:       {VALIDATE}")
    print(f"  Fast Mode:      {FAST_MODE}")
    print(f"  Tune Hyperpar:  {TUNE_HYPERPARAMS}")
    print(f"  GPU:            {torch.cuda.is_available()}")
    print(f"  Optuna:         {OPTUNA_AVAILABLE}")
    print(f"  Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
            fast_mode=FAST_MODE,
            tune_hyperparams=TUNE_HYPERPARAMS
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
        
        # Time budget check
        if total_time > 60:
            print(f"\nWARNING: Exceeded 60-minute target by {total_time-60:.1f} minutes")
            print("   Consider: Setting TUNE_HYPERPARAMS=False or FAST_MODE=True")
        else:
            print(f"\nSUCCESS: Completed within target ({60-total_time:.1f} min remaining)")

        submission.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSubmission saved: {OUTPUT_PATH}")
        print("="*60)
        print("\nOPTIMIZATIONS APPLIED:")
        print("="*60)
        print("  1. Optuna hyperparameter tuning (25 trials per model)")
        print("  2. Enhanced L1/L2 regularization")
        print("  3. CatBoost added to ensemble")
        print("  4. 3 new low-cost text features")
        print("  5. Bug fixes in TF-IDF transform")
        print("  6. Performance-based ensemble weighting")
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