"""
OPTIMIZED FAIL-PROOF Product Pricing Model (v4 - Anti-Overfitting)
- NEW: Aggressive regularization and simpler model architecture to fight overfitting
- NEW: Automatic Feature Selection to remove noise
- UPDATE: Switched to 5-Fold Cross-Validation for more robust validation
- All console output is saved to a detailed report file
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

# Logger Class to save all print statements to a file
class Logger(object):
    """
    A simple logger that writes to a file and the console simultaneously.
    This captures every print statement for a complete execution report.
    """
    def __init__(self, filename="training_report.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# Safe import of Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    print("INFO: Optuna loaded successfully")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: Optuna not available. Using default hyperparameters.")

sys.path.append('Data/student_resource')

# Safe import of utils.py
try:
    from src.utils import download_images
    UTILS_AVAILABLE = True
except ImportError:
    print("WARNING: utils.py not available. Image download will be disabled.")
    UTILS_AVAILABLE = False
    def download_images(links, path):
        raise ImportError("utils.py not available")

warnings.filterwarnings('ignore')

# Safe import of sentence-transformers
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
                self.tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 3), stop_words='english', min_df=2)
        
        if not self.use_embeddings:
            print("Initializing TF-IDF vectorizer...")
            self.tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 3), stop_words='english', min_df=2)
            print("   SUCCESS: TF-IDF ready")

    def extract_handcrafted_features(self, texts):
        features = {}
        try:
            features['text_length'] = [len(str(text)) for text in texts]
            features['word_count'] = [len(str(text).split()) for text in texts]
            features['avg_word_length'] = [np.mean([len(w) for w in str(text).split()]) if len(str(text).split()) > 0 else 0 for text in texts]
            features['punc_count'] = [len(re.findall(r'[!?,.]', str(text))) for text in texts]
            features['capital_ratio'] = [sum(1 for c in str(text) if c.isupper()) / len(str(text)) if len(str(text)) > 0 else 0 for text in texts]
            
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
            
            print(f"   Extracted {len(features)} handcrafted features")
        except Exception as e:
            print(f"   ERROR in handcrafted features: {e}")
            features = {'text_length': [len(str(text)) for text in texts]}
        return pd.DataFrame(features)

    def fit_transform(self, texts):
        handcrafted = self.extract_handcrafted_features(texts)
        if self.use_embeddings:
            try:
                print("   Generating text embeddings...")
                embeddings = self.text_model.encode([str(text) for text in texts], show_progress_bar=True, batch_size=128)
                embeddings_df = pd.DataFrame(embeddings, columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])])
                return pd.concat([handcrafted, embeddings_df], axis=1)
            except Exception as e:
                print(f"   ERROR in embeddings: {e}. Falling back to TF-IDF.")
                self.use_embeddings = False
                self.tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 3), stop_words='english', min_df=2)
        
        tfidf_features = self.tfidf.fit_transform([str(text) for text in texts])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=self.tfidf.get_feature_names_out())
        return pd.concat([handcrafted, tfidf_df], axis=1)

    def transform(self, texts):
        handcrafted = self.extract_handcrafted_features(texts)
        if self.use_embeddings:
            try:
                embeddings = self.text_model.encode([str(text) for text in texts], show_progress_bar=True, batch_size=128)
                embeddings_df = pd.DataFrame(embeddings, columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])])
                return pd.concat([handcrafted, embeddings_df], axis=1)
            except:
                return handcrafted
        else:
            tfidf_features = self.tfidf.transform([str(text) for text in texts])
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=self.tfidf.get_feature_names_out())
            return pd.concat([handcrafted, tfidf_df], axis=1)

class ImageFeatureExtractor:
    """FAIL-PROOF image feature extractor"""
    
    def __init__(self, image_dir='images'):
        self.image_dir = image_dir
        self.failed = True
        try:
            print("Loading ResNet50 model...")
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print(f"   SUCCESS: Model loaded on {self.device}")
            self.failed = False
        except Exception as e:
            print(f"   ERROR: Failed to load ResNet50: {e}. Image features will be zeros.")

    def extract_features(self, df, dataset_type='train', cache_file=None):
        if cache_file and os.path.exists(cache_file):
            try:
                cached = pd.read_pickle(cache_file)
                if len(cached) == len(df):
                    print(f"   Loading cached image features from {cache_file}")
                    return cached
            except: pass

        if self.failed:
            return pd.DataFrame(np.zeros((len(df), 2048)), columns=[f'img_feat_{i}' for i in range(2048)])

        features_list = []
        for idx in tqdm(range(len(df)), desc="   Processing images"):
            img_path = os.path.join(self.image_dir, dataset_type, f"{df.index[idx]}.jpg")
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model(img_tensor).squeeze().cpu().numpy()
            except:
                features = np.zeros(2048)
            features_list.append(features)
            
        image_df = pd.DataFrame(features_list, index=df.index, columns=[f'img_feat_{i}' for i in range(2048)])
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            image_df.to_pickle(cache_file)
        return image_df

class PricePredictionModel:
    """Model with anti-overfitting measures"""

    def __init__(self, use_images=True, use_embeddings=True, fast_mode=False, tune_hyperparams=True):
        self.use_images = use_images
        self.fast_mode = fast_mode
        self.tune_hyperparams = tune_hyperparams and OPTUNA_AVAILABLE
        self.text_extractor = TextFeatureExtractor(use_embeddings=use_embeddings)
        if use_images:
            self.image_extractor = ImageFeatureExtractor()
        self.scaler = RobustScaler()
        self.models = {}
        self.weights = {}
        self.best_params = {}
        self.selected_features = None

    def tune_hyperparameters(self, X_sample, y_sample):
        print("\n" + "="*60 + "\nHYPERPARAMETER TUNING (ANTI-OVERFITTING)\n" + "="*60)
        
        def lgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 350),
                'max_depth': trial.suggest_int('max_depth', 5, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 6.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 6.0),
                'random_state': 42, 'n_jobs': 1, 'verbose': -1
            }
            model = LGBMRegressor(**params)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = [mean_absolute_error(y_sample[val_idx], model.fit(X_sample[train_idx], y_sample[train_idx]).predict(X_sample[val_idx])) for train_idx, val_idx in kf.split(X_sample)]
            return np.mean(scores)

        print("[1/3] Tuning LightGBM...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lgb_objective, n_trials=30, show_progress_bar=False)
        self.best_params['lgb'] = study.best_params
        print(f"   Best LGB MAE: {study.best_value:.4f}")
        return self.best_params

    def _build_models(self, fast_mode=False):
        if fast_mode:
            return {'lgb': LGBMRegressor(n_estimators=150, random_state=42, n_jobs=-1, verbose=-1)}
        else:
            lgb_params = self.best_params.get('lgb', {})
            lgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
            return {'lgb': LGBMRegressor(**lgb_params)}

    def calculate_smape(self, y_true, y_pred):
        return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

    def fit(self, train_df, validate=True):
        print("\n" + "="*60 + "\nTRAINING PIPELINE (ANTI-OVERFITTING)\n" + "="*60)

        print("\n[1/6] Feature Extraction")
        X = self.text_extractor.fit_transform(train_df['catalog_content'].values)
        if self.use_images:
            X = pd.concat([X, self.image_extractor.extract_features(train_df, 'train', 'cache/train_image_features.pkl')], axis=1)
        
        y_log = np.log1p(train_df['price'].values)

        print("\n[2/6] Feature Scaling")
        X_scaled = self.scaler.fit_transform(X)

        print("\n[3/6] Feature Selection")
        fs_model = LGBMRegressor(n_estimators=150, random_state=42, n_jobs=-1, verbose=-1)
        fs_model.fit(X_scaled, y_log)
        importances = fs_model.feature_importances_
        self.selected_features = np.where(importances > np.mean(importances))[0]
        X_selected = X_scaled[:, self.selected_features]
        print(f"   Selected {len(self.selected_features)}/{X_scaled.shape[1]} features")

        if self.tune_hyperparams and not self.fast_mode:
            print("\n[4/6] Hyperparameter Tuning")
            self.best_params = self.tune_hyperparameters(X_selected[:30000], y_log[:30000])

        if validate:
            print(f"\n[5/6] Cross-Validation (5-Fold)")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected), 1):
                model = list(self._build_models(fast_mode=True).values())[0]
                model.fit(X_selected[train_idx], y_log[train_idx])
                pred = np.expm1(model.predict(X_selected[val_idx]))
                scores.append(self.calculate_smape(np.expm1(y_log[val_idx]), pred))
            print(f"   Mean SMAPE across 5 folds: {np.mean(scores):.2f}%")

        print(f"\n[6/6] Training Final Model")
        self.models = self._build_models(fast_mode=self.fast_mode)
        for model in self.models.values():
            model.fit(X_selected, y_log)

    def predict(self, test_df):
        print("\n" + "="*60 + "\nPREDICTION\n" + "="*60)
        X = self.text_extractor.transform(test_df['catalog_content'].values)
        if self.use_images:
            X = pd.concat([X, self.image_extractor.extract_features(test_df, 'test', 'cache/test_image_features.pkl')], axis=1)
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.selected_features]
        
        # Simplified prediction as we are using one model now
        final_model = list(self.models.values())[0]
        predictions_log = final_model.predict(X_selected)
        return np.maximum(np.expm1(predictions_log), 0.01)

def main():
    original_stdout = sys.stdout
    os.makedirs('reports', exist_ok=True)
    log_file = f"reports/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Logger(log_file)
    
    try:
        # Configuration
        USE_IMAGES = True
        USE_EMBEDDINGS = True
        VALIDATE = True
        FAST_MODE = False
        TUNE_HYPERPARAMS = True
        
        TRAIN_PATH = 'Data/student_resource/dataset/train.csv'
        TEST_PATH = 'Data/student_resource/dataset/test.csv'
        OUTPUT_PATH = 'submissions/test_out_anti_overfit.csv'
        MODEL_PATH = 'models/model_anti_overfit.pkl'

        for directory in ['cache', 'models', 'submissions', 'reports']:
            os.makedirs(directory, exist_ok=True)

        print("\n" + "="*60 + "\nFAIL-PROOF MODEL (v4 - Anti-Overfitting)\n" + "="*60)
        
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        
        start_time = time.time()
        
        model = PricePredictionModel(
            use_images=USE_IMAGES,
            use_embeddings=USE_EMBEDDINGS,
            fast_mode=FAST_MODE,
            tune_hyperparams=TUNE_HYPERPARAMS
        )
        model.fit(train_df, validate=VALIDATE)
        
        with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
        print(f"\nModel saved: {MODEL_PATH}")

        predictions = model.predict(test_df)
        
        submission = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': predictions})
        submission.to_csv(OUTPUT_PATH, index=False)
        
        total_time = (time.time() - start_time) / 60
        print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
        print(submission['price'].describe())
        print(f"\nTotal Pipeline Time: {total_time:.1f} minutes")
        print(f"Submission saved: {OUTPUT_PATH}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        traceback.print_exc()
    finally:
        if isinstance(sys.stdout, Logger):
            sys.stdout.log.close()
        sys.stdout = original_stdout
        print(f"\nLog file saved to: {log_file}")

if __name__ == "__main__":
    main()