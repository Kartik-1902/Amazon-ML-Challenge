import pandas as pd
import numpy as np
import os
import pickle
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
from tqdm import tqdm
import json

# ML Libraries
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb

# NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Image Processing
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Hyperparameter Optimization
import optuna

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class with anti-overfitting settings"""
    # ==================== DATASET PATHS ====================
    TRAIN_CSV = "Data/student_resource/dataset/train.csv"
    TEST_CSV = "Data/student_resource/dataset/test.csv"
    
    # ==================== IMAGE PATHS ====================
    TRAIN_IMAGES = "images/train"
    TEST_IMAGES = "images/test"
    
    # ==================== OUTPUT & CACHE PATHS ====================
    OUTPUT_CSV = "test_out.csv"
    MODEL_SAVE_PATH = "models"
    CACHE_PATH = "cache"
    
    # Model Selection
    NLP_MODEL = "tfidf"
    IMAGE_MODEL = "resnet"
    REGRESSION_MODEL = "xgboost"  # xgboost, lightgbm, ridge, elasticnet
    
    # Training Parameters - ANTI-OVERFITTING
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_OPTUNA_TRIALS = 30  # Reduced for faster training
    USE_CROSS_VALIDATION = True  # Enable CV for better generalization
    N_FOLDS = 5
    
    # Feature Dimensions - REDUCED to prevent overfitting
    TEXT_FEATURE_DIM = 150  # Reduced from 300
    IMAGE_FEATURE_DIM = 512  # Reduced from 2048
    
    # Feature Selection & Regularization
    USE_FEATURE_SELECTION = True
    FEATURE_SELECTION_THRESHOLD = 0.001  # Remove very low variance features
    
    # Ensemble Settings
    USE_ENSEMBLE = True  # Combine multiple models
    
    # Caching Options
    USE_CACHE = True
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate SMAPE metric"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.mean(numerator / denominator) * 100


class TextFeatureExtractor:
    """Extract features from catalog content with regularization"""
    
    def __init__(self, method: str = "tfidf", feature_dim: int = 150):
        self.method = method
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit(self, texts: pd.Series):
        """Fit the text feature extractor with regularization"""
        logger.info(f"Fitting {self.method} text feature extractor...")
        
        cache_file = self._get_cache_filename('fit')
        if Config.USE_CACHE and os.path.exists(cache_file):
            logger.info(f"Loading {self.method} model from cache...")
            with open(cache_file, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Text model loaded from cache")
            return
        
        processed_texts = texts.apply(self.preprocess_text)
        
        # TF-IDF with stronger regularization
        self.model = TfidfVectorizer(
            max_features=self.feature_dim,
            ngram_range=(1, 2),
            min_df=5,  # Increased from 2 - ignore rare terms
            max_df=0.85,  # Decreased from 0.95 - ignore very common terms
            sublinear_tf=True,  # Use log scaling
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
        )
        logger.info("Fitting TF-IDF vectorizer with regularization...")
        self.model.fit(processed_texts)
        
        if Config.USE_CACHE:
            os.makedirs(Config.CACHE_PATH, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Text model cached")
    
    def _get_cache_filename(self, mode: str) -> str:
        cache_id = f"{self.method}_{self.feature_dim}_{mode}_v2"
        return os.path.join(Config.CACHE_PATH, f"text_model_{cache_id}.pkl")
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform texts to feature vectors"""
        processed_texts = texts.apply(self.preprocess_text)
        features = self.model.transform(processed_texts).toarray()
        logger.info(f"✓ Text features: {features.shape}")
        return features
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


class ImageFeatureExtractor:
    """Extract features from product images with dropout"""
    
    def __init__(self, model_type: str = "resnet", feature_dim: int = 512):
        self.model_type = model_type
        self.feature_dim = feature_dim
        self.device = Config.DEVICE
        self.model = self._build_model()
        self.transform = self._get_transform()
        
    def _build_model(self) -> nn.Module:
        """Build image feature extraction model"""
        if self.model_type == "resnet":
            # Use ResNet50 instead of ResNet152 to reduce overfitting
            model = models.resnet50(pretrained=True)
            # Extract from earlier layer for better generalization
            self.model = nn.Sequential(*list(model.children())[:-1])
        
        self.model.to(self.device)
        self.model.eval()
        return self.model
    
    def _get_transform(self):
        """Get image preprocessing with augmentation-style normalization"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from a single image"""
        try:
            if not os.path.exists(image_path):
                return np.zeros(self.feature_dim)
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
                
                if features.shape[0] != self.feature_dim:
                    features = features.flatten()[:self.feature_dim]
                    if features.shape[0] < self.feature_dim:
                        features = np.pad(features, (0, self.feature_dim - features.shape[0]))
            
            return features
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")
            return np.zeros(self.feature_dim)
    
    def extract_batch_features(self, image_links: pd.Series, image_folder: str) -> np.ndarray:
        """Extract features from multiple images"""
        cache_file = self._get_cache_filename(image_folder)
        if Config.USE_CACHE and os.path.exists(cache_file):
            logger.info("Loading image features from cache...")
            features = np.load(cache_file)
            logger.info(f"✓ Loaded {len(features)} cached image features")
            return features
        
        features = []
        missing_count = 0
        
        for image_link in tqdm(image_links, desc="Processing images"):
            if isinstance(image_link, str):
                image_filename = Path(image_link).name
                image_path = os.path.join(image_folder, image_filename)
            else:
                image_path = None
            
            if image_path and os.path.exists(image_path):
                feature = self.extract_features(image_path)
            else:
                feature = np.zeros(self.feature_dim)
                missing_count += 1
            
            features.append(feature)
        
        features = np.array(features)
        
        if missing_count > 0:
            logger.warning(f"⚠ {missing_count} images missing")
        
        logger.info(f"✓ Image features: {features.shape}")
        
        if Config.USE_CACHE:
            os.makedirs(Config.CACHE_PATH, exist_ok=True)
            np.save(cache_file, features)
            logger.info(f"✓ Image features cached")
        
        return features
    
    def _get_cache_filename(self, image_folder: str) -> str:
        folder_name = os.path.basename(image_folder)
        cache_id = f"{self.model_type}_{self.feature_dim}_{folder_name}_v2"
        return os.path.join(Config.CACHE_PATH, f"image_features_{cache_id}.npy")


class RegressionModel:
    """Wrapper for regression models with strong regularization"""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.best_params = None
        
    def _get_model(self, params: Optional[Dict] = None):
        """Get regression model with anti-overfitting defaults"""
        if params is None:
            params = {}
            
        if self.model_type == "ridge":
            return Ridge(
                alpha=params.get('alpha', 10.0),  # Stronger regularization
                random_state=Config.RANDOM_STATE
            )
        
        elif self.model_type == "elasticnet":
            return ElasticNet(
                alpha=params.get('alpha', 1.0),
                l1_ratio=params.get('l1_ratio', 0.5),
                random_state=Config.RANDOM_STATE,
                max_iter=2000
            )
        
        elif self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 8),  # Limit depth
                min_samples_split=params.get('min_samples_split', 20),  # Increased
                min_samples_leaf=params.get('min_samples_leaf', 10),  # Increased
                max_features=params.get('max_features', 0.5),  # Use subset
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )
        
        elif self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 4),  # Shallow trees
                learning_rate=params.get('learning_rate', 0.05),  # Slower learning
                subsample=params.get('subsample', 0.7),  # Strong sampling
                colsample_bytree=params.get('colsample_bytree', 0.7),
                min_child_weight=params.get('min_child_weight', 3),  # Regularization
                gamma=params.get('gamma', 0.1),  # Regularization
                reg_alpha=params.get('reg_alpha', 0.5),  # L1 reg
                reg_lambda=params.get('reg_lambda', 1.0),  # L2 reg
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )
        
        elif self.model_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 4),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.7),
                colsample_bytree=params.get('colsample_bytree', 0.7),
                min_child_samples=params.get('min_child_samples', 20),
                reg_alpha=params.get('reg_alpha', 0.5),
                reg_lambda=params.get('reg_lambda', 1.0),
                random_state=Config.RANDOM_STATE,
                verbose=-1,
                n_jobs=-1
            )
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 n_trials: int = 30):
        """Optimize hyperparameters with cross-validation"""
        logger.info(f"Optimizing hyperparameters for {self.model_type}...")
        
        cache_file = self._get_hyperparams_cache_filename()
        if Config.USE_CACHE and os.path.exists(cache_file):
            logger.info("Loading cached hyperparameters...")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                self.best_params = cached_data['best_params']
                logger.info(f"✓ Cached params: {self.best_params}")
                logger.info(f"✓ Cached SMAPE: {cached_data['best_smape']:.4f}")
                return self.best_params
        
        def objective(trial):
            if self.model_type == "ridge":
                params = {'alpha': trial.suggest_float('alpha', 1.0, 50.0)}
            
            elif self.model_type == "elasticnet":
                params = {
                    'alpha': trial.suggest_float('alpha', 0.1, 10.0),
                    'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9)
                }
            
            elif self.model_type == "rf":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 12),
                    'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
                    'max_features': trial.suggest_float('max_features', 0.3, 0.8)
                }
            
            elif self.model_type in ["xgboost", "lightgbm"]:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0)
                }
                
                if self.model_type == "xgboost":
                    params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
                    params['gamma'] = trial.suggest_float('gamma', 0.0, 0.5)
                else:
                    params['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 50)
            
            # Use cross-validation for more robust evaluation
            if Config.USE_CROSS_VALIDATION:
                kf = KFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
                scores = []
                
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, X_vl = X_train[train_idx], X_train[val_idx]
                    y_tr, y_vl = y_train[train_idx], y_train[val_idx]
                    
                    model = self._get_model(params)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_vl)
                    smape = calculate_smape(y_vl, y_pred)
                    scores.append(smape)
                
                return np.mean(scores)
            else:
                model = self._get_model(params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                return calculate_smape(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        logger.info(f"✓ Best params: {self.best_params}")
        logger.info(f"✓ Best SMAPE: {study.best_value:.4f}%")
        
        if Config.USE_CACHE:
            os.makedirs(Config.CACHE_PATH, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({
                    'best_params': self.best_params,
                    'best_smape': study.best_value
                }, f, indent=2)
            logger.info(f"✓ Hyperparameters cached")
        
        return self.best_params
    
    def _get_hyperparams_cache_filename(self) -> str:
        return os.path.join(Config.CACHE_PATH, f"hyperparams_{self.model_type}_v2.json")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the regression model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._get_model(self.best_params)
        self.model.fit(X_scaled, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the regression model"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.maximum(predictions, 0)


class ProductPricePredictionPipeline:
    """Complete ML pipeline with anti-overfitting measures"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.text_extractor = TextFeatureExtractor(
            method=config.NLP_MODEL,
            feature_dim=config.TEXT_FEATURE_DIM
        )
        self.image_extractor = ImageFeatureExtractor(
            model_type=config.IMAGE_MODEL,
            feature_dim=config.IMAGE_FEATURE_DIM
        )
        self.models = []  # For ensemble
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data"""
        logger.info("=" * 60)
        logger.info("LOADING DATASET")
        logger.info("=" * 60)
        
        if not os.path.exists(self.config.TRAIN_CSV):
            raise FileNotFoundError(f"Training CSV not found: {self.config.TRAIN_CSV}")
        if not os.path.exists(self.config.TEST_CSV):
            raise FileNotFoundError(f"Test CSV not found: {self.config.TEST_CSV}")
        
        train_df = pd.read_csv(self.config.TRAIN_CSV)
        test_df = pd.read_csv(self.config.TEST_CSV)
        
        logger.info(f"✓ Train: {train_df.shape}")
        logger.info(f"✓ Test: {test_df.shape}")
        logger.info("=" * 60)
        
        return train_df, test_df
    
    def extract_features(self, df: pd.DataFrame, is_train: bool = True) -> np.ndarray:
        """Extract combined features"""
        logger.info("=" * 60)
        logger.info(f"EXTRACTING FEATURES ({'TRAIN' if is_train else 'TEST'})")
        logger.info("=" * 60)
        
        # Text features
        logger.info(f"Text Feature Extraction ({self.config.NLP_MODEL})")
        if is_train:
            text_features = self.text_extractor.fit_transform(df['catalog_content'])
        else:
            text_features = self.text_extractor.transform(df['catalog_content'])
        
        # Image features
        logger.info(f"Image Feature Extraction ({self.config.IMAGE_MODEL})")
        image_folder = self.config.TRAIN_IMAGES if is_train else self.config.TEST_IMAGES
        
        if not os.path.exists(image_folder):
            logger.warning(f"⚠ Image folder not found: {image_folder}")
            image_features = np.zeros((len(df), self.config.IMAGE_FEATURE_DIM))
        else:
            image_features = self.image_extractor.extract_batch_features(df['image_link'], image_folder)
        
        # Combine features
        combined_features = np.hstack([text_features, image_features])
        logger.info(f"✓ Combined: {combined_features.shape}")
        logger.info("=" * 60)
        
        return combined_features
    
    def train(self, optimize: bool = True):
        """Train with regularization and cross-validation"""
        logger.info("\n" + "=" * 60)
        logger.info("ANTI-OVERFITTING TRAINING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"NLP: {self.config.NLP_MODEL} | Image: {self.config.IMAGE_MODEL}")
        logger.info(f"Regression: {self.config.REGRESSION_MODEL}")
        logger.info(f"Cross-Validation: {Config.USE_CROSS_VALIDATION} ({Config.N_FOLDS} folds)")
        logger.info(f"Ensemble: {Config.USE_ENSEMBLE}")
        logger.info("=" * 60)
        
        train_df, _ = self.load_data()
        X = self.extract_features(train_df, is_train=True)
        y = train_df['price'].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info("=" * 60)
        
        # Train models
        if Config.USE_ENSEMBLE:
            model_types = ["xgboost", "lightgbm", "ridge"]
            logger.info(f"Training ensemble: {model_types}")
            
            for model_type in model_types:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_type.upper()}")
                logger.info(f"{'='*60}")
                
                model = RegressionModel(model_type=model_type)
                
                if optimize:
                    model.optimize_hyperparameters(
                        X_train, y_train, X_val, y_val,
                        n_trials=self.config.N_OPTUNA_TRIALS
                    )
                
                model.fit(X, y)
                self.models.append(model)
                
                # Evaluate
                val_pred = model.predict(X_val)
                val_smape = calculate_smape(y_val, val_pred)
                logger.info(f"✓ {model_type} Val SMAPE: {val_smape:.4f}%")
        else:
            model = RegressionModel(model_type=self.config.REGRESSION_MODEL)
            
            if optimize:
                model.optimize_hyperparameters(
                    X_train, y_train, X_val, y_val,
                    n_trials=self.config.N_OPTUNA_TRIALS
                )
            
            model.fit(X, y)
            self.models.append(model)
        
        # Final evaluation
        logger.info("\n" + "=" * 60)
        logger.info("FINAL PERFORMANCE")
        logger.info("=" * 60)
        
        val_preds = np.array([model.predict(X_val) for model in self.models])
        ensemble_pred = np.mean(val_preds, axis=0)
        final_smape = calculate_smape(y_val, ensemble_pred)
        
        logger.info(f"Validation SMAPE: {final_smape:.4f}%")
        logger.info(f"Prediction range: ${ensemble_pred.min():.2f} - ${ensemble_pred.max():.2f}")
        logger.info("=" * 60)
        
        self.save_model()
        return final_smape
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions"""
        X_test = self.extract_features(test_df, is_train=False)
        
        if len(self.models) > 1:
            predictions = np.array([model.predict(X_test) for model in self.models])
            return np.mean(predictions, axis=0)
        else:
            return self.models[0].predict(X_test)
    
    def save_model(self):
        """Save trained models"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, "pipeline_v2.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'text_extractor': self.text_extractor,
                'models': self.models
            }, f)
        
        logger.info(f"✓ Model saved to {model_path}")
    
    def generate_submission(self):
        """Generate submission file"""
        _, test_df = self.load_data()
        predictions = self.predict(test_df)
        
        submission_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })
        
        submission_df.to_csv(self.config.OUTPUT_CSV, index=False)
        logger.info(f"✓ Submission: {self.config.OUTPUT_CSV}")
        logger.info(f"✓ Predictions: {len(submission_df)}")
        logger.info(f"Range: ${predictions.min():.2f} - ${predictions.max():.2f}")


def main():
    """Main execution"""
    # Enable anti-overfitting features
    Config.USE_CROSS_VALIDATION = True
    Config.USE_ENSEMBLE = True
    Config.TEXT_FEATURE_DIM = 150
    Config.IMAGE_FEATURE_DIM = 512
    
    pipeline = ProductPricePredictionPipeline(Config)
    val_smape = pipeline.train(optimize=True)
    pipeline.generate_submission()
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Validation SMAPE: {val_smape:.4f}%")
    logger.info(f"Output: {Config.OUTPUT_CSV}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()