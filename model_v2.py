import pandas as pd
import numpy as np
import os
import pickle
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
from tqdm import tqdm
import hashlib
import json

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb

# NLP Libraries
from gensim.models import Word2Vec
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
    """Configuration class for model parameters"""
    # ==================== DATASET PATHS ====================
    TRAIN_CSV = "Data/student_resource/dataset/train.csv"
    TEST_CSV = "Data/student_resource/dataset/test.csv"
    
    # ==================== IMAGE PATHS ====================
    TRAIN_IMAGES = "images/train"
    TEST_IMAGES = "images/test"
    
    # ==================== OUTPUT & CACHE PATHS ====================
    OUTPUT_CSV = "test_out.csv"
    MODEL_SAVE_PATH = "models"
    CACHE_PATH = "cache_testing"
    CACHE_PATH_HP = "cache"
    
    # Model Selection
    NLP_MODEL = "glove"  # Options: "word2vec", "glove", "tfidf"
    IMAGE_MODEL = "resnet"  # Options: "resnet", "cnn"
    REGRESSION_MODEL = "xgboost"  # Options: "linear", "ridge", "lasso", "svm", "rf", "xgboost", "lightgbm"
    
    # Training Parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_OPTUNA_TRIALS = 50
    
    # Feature Dimensions
    TEXT_FEATURE_DIM = 100
    IMAGE_FEATURE_DIM = 128
    
    # Caching Options
    USE_CACHE = True
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate SMAPE metric"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100


class TextFeatureExtractor:
    """Extract features from catalog content"""
    
    def __init__(self, method: str = "tfidf", feature_dim: int = 300):
        self.method = method
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit(self, texts: pd.Series):
        """Fit the text feature extractor"""
        logger.info(f"Fitting {self.method} text feature extractor...")
        
        # Check cache
        cache_file = self._get_cache_filename('fit')
        if Config.USE_CACHE and os.path.exists(cache_file):
            logger.info(f"Loading {self.method} model from cache...")
            with open(cache_file, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("✓ Text model loaded from cache")
            return
        
        processed_texts = texts.apply(self.preprocess_text)
        logger.info(f"Preprocessing {len(texts)} text samples...")
        
        if self.method == "tfidf":
            self.model = TfidfVectorizer(
                max_features=self.feature_dim,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            logger.info("Fitting TF-IDF vectorizer...")
            self.model.fit(processed_texts)
            
        elif self.method == "word2vec":
            tokenized_texts = [text.split() for text in tqdm(processed_texts, desc="Tokenizing texts") if text]
            logger.info(f"Training Word2Vec model on {len(tokenized_texts)} documents...")
            self.model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.feature_dim,
                window=5,
                min_count=2,
                workers=4,
                epochs=10
            )
        
        elif self.method == "glove":
            logger.info("Fitting GloVe-style vectorizer...")
            self.model = TfidfVectorizer(
                max_features=self.feature_dim,
                ngram_range=(1, 3),
                min_df=2
            )
            self.model.fit(processed_texts)
        
        # Save to cache
        if Config.USE_CACHE:
            os.makedirs(Config.CACHE_PATH, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"✓ Text model cached to {cache_file}")
    
    def _get_cache_filename(self, mode: str) -> str:
        """Generate cache filename based on method and parameters"""
        cache_id = f"{self.method}_{self.feature_dim}_{mode}"
        return os.path.join(Config.CACHE_PATH, f"text_model_{cache_id}.pkl")
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform texts to feature vectors"""
        logger.info(f"Transforming {len(texts)} text samples...")
        processed_texts = texts.apply(self.preprocess_text)
        
        if self.method == "tfidf" or self.method == "glove":
            features = self.model.transform(processed_texts).toarray()
            
        elif self.method == "word2vec":
            features = []
            for text in tqdm(processed_texts, desc="Generating embeddings"):
                if not text:
                    features.append(np.zeros(self.feature_dim))
                    continue
                words = text.split()
                word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
                if word_vectors:
                    features.append(np.mean(word_vectors, axis=0))
                else:
                    features.append(np.zeros(self.feature_dim))
            features = np.array(features)
        
        logger.info(f"✓ Text features generated: {features.shape}")
        return features
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Fit and transform texts"""
        self.fit(texts)
        return self.transform(texts)


class ImageFeatureExtractor:
    """Extract features from product images"""
    
    def __init__(self, model_type: str = "resnet", feature_dim: int = 512):
        self.model_type = model_type
        self.feature_dim = feature_dim
        self.device = Config.DEVICE
        self.model = self._build_model()
        self.transform = self._get_transform()
        
    def _build_model(self) -> nn.Module:
        """Build image feature extraction model"""
        if self.model_type == "resnet":
            model = models.resnet50(pretrained=True)
            # Remove final classification layer
            self.model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_type == "cnn":
            # Simple CNN architecture
            self.model = SimpleCNN(output_dim=self.feature_dim)
        
        self.model.to(self.device)
        self.model.eval()
        return self.model
    
    def _get_transform(self):
        """Get image preprocessing transform"""
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
        """Extract features from multiple images - FIXED to use original filenames"""
        logger.info(f"Extracting image features from {image_folder}...")
        
        # Check cache
        cache_file = self._get_cache_filename(image_folder)
        if Config.USE_CACHE and os.path.exists(cache_file):
            logger.info("Loading image features from cache...")
            features = np.load(cache_file)
            logger.info(f"✓ Loaded {len(features)} cached image features")
            return features
        
        features = []
        missing_count = 0
        
        # FIXED: Use image_link to get original filename
        for image_link in tqdm(image_links, desc="Processing images"):
            if isinstance(image_link, str):
                # Get original filename from URL
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
            logger.warning(f"⚠ {missing_count} images were missing (set to zero vectors)")
        else:
            logger.info(f"✓ All images found!")
        
        logger.info(f"✓ Image features extracted: {features.shape}")
        
        # Save to cache
        if Config.USE_CACHE:
            os.makedirs(Config.CACHE_PATH, exist_ok=True)
            np.save(cache_file, features)
            logger.info(f"✓ Image features cached to {cache_file}")
        
        return features
    
    def _get_cache_filename(self, image_folder: str) -> str:
        """Generate cache filename based on model type and image folder"""
        folder_name = os.path.basename(image_folder)
        cache_id = f"{self.model_type}_{self.feature_dim}_{folder_name}"
        return os.path.join(Config.CACHE_PATH, f"image_features_{cache_id}.npy")


class SimpleCNN(nn.Module):
    """Simple CNN for image feature extraction"""
    
    def __init__(self, output_dim: int = 512):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RegressionModel:
    """Wrapper for various regression models"""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        
    def _get_model(self, params: Optional[Dict] = None):
        """Get regression model based on type"""
        if params is None:
            params = {}
            
        if self.model_type == "linear":
            return Ridge(alpha=params.get('alpha', 1.0))
        
        elif self.model_type == "ridge":
            return Ridge(alpha=params.get('alpha', 1.0))
        
        elif self.model_type == "lasso":
            return Lasso(alpha=params.get('alpha', 1.0))
        
        elif self.model_type == "svm":
            return SVR(
                C=params.get('C', 1.0),
                epsilon=params.get('epsilon', 0.1),
                kernel=params.get('kernel', 'rbf')
            )
        
        elif self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2),
                random_state=Config.RANDOM_STATE
            )
        
        elif self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=Config.RANDOM_STATE
            )
        
        elif self.model_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=Config.RANDOM_STATE,
                verbose=-1
            )
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 n_trials: int = 50):
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Optimizing hyperparameters for {self.model_type}...")
        logger.info(f"Running {n_trials} trials with Optuna...")
        
        # Check cache
        cache_file = self._get_hyperparams_cache_filename()
        if Config.USE_CACHE and os.path.exists(cache_file):
            logger.info("Loading cached hyperparameters...")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                self.best_params = cached_data['best_params']
                logger.info(f"✓ Loaded cached parameters: {self.best_params}")
                logger.info(f"✓ Cached SMAPE: {cached_data['best_smape']:.4f}")
                return self.best_params
        
        def objective(trial):
            if self.model_type in ["linear", "ridge"]:
                params = {'alpha': trial.suggest_float('alpha', 0.01, 10.0)}
            
            elif self.model_type == "lasso":
                params = {'alpha': trial.suggest_float('alpha', 0.01, 10.0)}
            
            elif self.model_type == "svm":
                params = {
                    'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear'])
                }
            
            elif self.model_type == "rf":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                }
            
            elif self.model_type in ["xgboost", "lightgbm"]:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            
            model = self._get_model(params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            smape = calculate_smape(y_val, y_pred)
            
            return smape
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        logger.info(f"✓ Best parameters: {self.best_params}")
        logger.info(f"✓ Best SMAPE: {study.best_value:.4f}%")
        
        # Cache the results
        if Config.USE_CACHE:
            os.makedirs(Config.CACHE_PATH, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({
                    'best_params': self.best_params,
                    'best_smape': study.best_value
                }, f, indent=2)
            logger.info(f"✓ Hyperparameters cached to {cache_file}")
        
        return self.best_params
    
    def _get_hyperparams_cache_filename(self) -> str:
        """Generate cache filename for hyperparameters"""
        return os.path.join(Config.CACHE_PATH_HP, f"hyperparams_{self.model_type}.json")
    
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
    """Complete ML pipeline for product price prediction"""
    
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
        self.regression_model = RegressionModel(model_type=config.REGRESSION_MODEL)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data"""
        logger.info("=" * 60)
        logger.info("LOADING DATASET")
        logger.info("=" * 60)
        logger.info(f"Train CSV path: {self.config.TRAIN_CSV}")
        logger.info(f"Test CSV path:  {self.config.TEST_CSV}")
        
        if not os.path.exists(self.config.TRAIN_CSV):
            raise FileNotFoundError(f"Training CSV not found at: {self.config.TRAIN_CSV}")
        if not os.path.exists(self.config.TEST_CSV):
            raise FileNotFoundError(f"Test CSV not found at: {self.config.TEST_CSV}")
        
        train_df = pd.read_csv(self.config.TRAIN_CSV)
        test_df = pd.read_csv(self.config.TEST_CSV)
        
        logger.info(f"✓ Train data loaded: {train_df.shape}")
        logger.info(f"✓ Test data loaded:  {test_df.shape}")
        logger.info(f"Train columns: {list(train_df.columns)}")
        logger.info(f"Test columns:  {list(test_df.columns)}")
        logger.info("=" * 60)
        
        return train_df, test_df
    
    def extract_features(self, df: pd.DataFrame, is_train: bool = True) -> np.ndarray:
        """Extract combined text and image features - FIXED to use image_link"""
        logger.info("=" * 60)
        logger.info(f"EXTRACTING FEATURES ({'TRAIN' if is_train else 'TEST'})")
        logger.info("=" * 60)
        
        # Text features
        logger.info(f"Step 1/2: Text Feature Extraction ({self.config.NLP_MODEL})")
        if is_train:
            text_features = self.text_extractor.fit_transform(df['catalog_content'])
        else:
            text_features = self.text_extractor.transform(df['catalog_content'])
        
        # Image features - FIXED: Pass image_link instead of sample_id
        logger.info(f"Step 2/2: Image Feature Extraction ({self.config.IMAGE_MODEL})")
        image_folder = self.config.TRAIN_IMAGES if is_train else self.config.TEST_IMAGES
        logger.info(f"Image folder path: {image_folder}")
        
        if not os.path.exists(image_folder):
            logger.warning(f"⚠ Image folder not found: {image_folder}")
            logger.warning(f"⚠ Using zero vectors for all image features!")
            image_features = np.zeros((len(df), self.config.IMAGE_FEATURE_DIM))
        else:
            # FIXED: Pass image_link column instead of sample_id
            image_features = self.image_extractor.extract_batch_features(df['image_link'], image_folder)
        
        # Combine features
        combined_features = np.hstack([text_features, image_features])
        logger.info(f"✓ Combined feature shape: {combined_features.shape}")
        logger.info(f"  - Text features:  {text_features.shape}")
        logger.info(f"  - Image features: {image_features.shape}")
        logger.info("=" * 60)
        
        return combined_features
    
    def train(self, optimize: bool = True):
        """Train the complete pipeline"""
        logger.info("\n")
        logger.info("=" * 60)
        logger.info("🚀 STARTING TRAINING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  - NLP Model:        {self.config.NLP_MODEL}")
        logger.info(f"  - Image Model:      {self.config.IMAGE_MODEL}")
        logger.info(f"  - Regression Model: {self.config.REGRESSION_MODEL}")
        logger.info(f"  - Device:           {self.config.DEVICE}")
        logger.info(f"  - Caching:          {'Enabled' if self.config.USE_CACHE else 'Disabled'}")
        logger.info(f"  - Optuna Trials:    {self.config.N_OPTUNA_TRIALS}")
        logger.info("=" * 60)
        
        # Load data
        train_df, _ = self.load_data()
        
        # Extract features
        X = self.extract_features(train_df, is_train=True)
        y = train_df['price'].values
        
        logger.info("=" * 60)
        logger.info("TRAIN/VALIDATION SPLIT")
        logger.info("=" * 60)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        logger.info(f"✓ Train samples:      {len(X_train):,}")
        logger.info(f"✓ Validation samples: {len(X_val):,}")
        logger.info(f"✓ Feature dimension:  {X_train.shape[1]}")
        logger.info(f"Price statistics:")
        logger.info(f"  - Min:  ${y_train.min():.2f}")
        logger.info(f"  - Max:  ${y_train.max():.2f}")
        logger.info(f"  - Mean: ${y_train.mean():.2f}")
        logger.info("=" * 60)
        
        # Optimize hyperparameters
        if optimize:
            logger.info("=" * 60)
            logger.info("HYPERPARAMETER OPTIMIZATION")
            logger.info("=" * 60)
            self.regression_model.optimize_hyperparameters(
                X_train, y_train, X_val, y_val,
                n_trials=self.config.N_OPTUNA_TRIALS
            )
            logger.info("=" * 60)
        
        # Train final model
        logger.info("=" * 60)
        logger.info("TRAINING FINAL MODEL")
        logger.info("=" * 60)
        logger.info(f"Training {self.config.REGRESSION_MODEL} on full training set...")
        self.regression_model.fit(X, y)
        logger.info("✓ Model training completed")
        
        # Evaluate
        logger.info("\nEvaluating model performance...")
        train_pred = self.regression_model.predict(X_train)
        val_pred = self.regression_model.predict(X_val)
        
        train_smape = calculate_smape(y_train, train_pred)
        val_smape = calculate_smape(y_val, val_pred)
        
        logger.info("=" * 60)
        logger.info("📊 MODEL PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Train SMAPE: {train_smape:.4f}%")
        logger.info(f"Val SMAPE:   {val_smape:.4f}%")
        logger.info(f"\nPrediction Statistics (Validation Set):")
        logger.info(f"  - Min:  ${val_pred.min():.2f}")
        logger.info(f"  - Max:  ${val_pred.max():.2f}")
        logger.info(f"  - Mean: ${val_pred.mean():.2f}")
        logger.info("=" * 60)
        
        # Save model
        logger.info("\nSaving model...")
        self.save_model()
        
        return train_smape, val_smape
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for test data"""
        logger.info("Generating predictions...")
        X_test = self.extract_features(test_df, is_train=False)
        predictions = self.regression_model.predict(X_test)
        return predictions
    
    def save_model(self):
        """Save trained models"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, "pipeline.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'text_extractor': self.text_extractor,
                'regression_model': self.regression_model
            }, f)
        
        logger.info(f"✓ Model saved to {model_path}")
    
    def load_model(self):
        """Load trained models"""
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, "pipeline.pkl")
        
        with open(model_path, 'rb') as f:
            saved_models = pickle.load(f)
            self.text_extractor = saved_models['text_extractor']
            self.regression_model = saved_models['regression_model']
        
        logger.info(f"Model loaded from {model_path}")
    
    def generate_submission(self):
        """Generate submission file"""
        _, test_df = self.load_data()
        predictions = self.predict(test_df)
        
        submission_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })
        
        submission_df.to_csv(self.config.OUTPUT_CSV, index=False)
        logger.info(f"✓ Submission file saved to {self.config.OUTPUT_CSV}")
        logger.info(f"✓ Total predictions: {len(submission_df)}")
        logger.info(f"Prediction stats - Min: ${predictions.min():.2f}, "
                   f"Max: ${predictions.max():.2f}, Mean: ${predictions.mean():.2f}")


def main():
    """Main execution function"""
    # Configuration
    Config.NLP_MODEL = "tfidf"  # Options: "word2vec", "glove", "tfidf"
    Config.IMAGE_MODEL = "resnet"  # Options: "resnet", "cnn"
    Config.REGRESSION_MODEL = "xgboost"  # Options: "linear", "ridge", "lasso", "svm", "rf", "xgboost", "lightgbm"
    Config.N_OPTUNA_TRIALS = 50
    
    # Initialize pipeline
    pipeline = ProductPricePredictionPipeline(Config)
    
    # Train model
    train_smape, val_smape = pipeline.train(optimize=True)
    
    # Generate submission
    pipeline.generate_submission()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Final Validation SMAPE: {val_smape:.4f}%")
    logger.info(f"Submission file: {Config.OUTPUT_CSV}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()