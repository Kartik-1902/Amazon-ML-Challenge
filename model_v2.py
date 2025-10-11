"""
FIXED ML Pipeline with Proper Progress Tracking
Key fix: Manual K-Fold with real-time fold progress (no more waiting blind!)
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
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
import re
import warnings
import pickle
import os
from tqdm import tqdm
import sys
from datetime import datetime
import time

sys.path.append('Data/student_resource')
from src.utils import download_images

warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor:
    """Extract features from catalog content"""

    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE

        if self.use_embeddings:
            print("📝 Loading text embedding model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("   ✅ Model loaded")
        else:
            self.tfidf = TfidfVectorizer(
                max_features=1500,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2
            )

    def extract_handcrafted_features(self, texts):
        """Extract text features"""
        features = {}

        features['text_length'] = [len(str(text)) for text in texts]
        features['word_count'] = [len(str(text).split()) for text in texts]
        features['avg_word_length'] = [
            np.mean([len(word) for word in str(text).split()]) if len(str(text).split()) > 0 else 0
            for text in texts
        ]
        features['unique_chars'] = [len(set(str(text))) for text in texts]
        features['sentence_count'] = [len(str(text).split('.')) for text in texts]

        # Extract IPQ
        ipq_values = []
        for text in texts:
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
        features['ipq'] = ipq_values
        features['ipq_log'] = np.log1p(ipq_values)

        # Keywords
        premium_keywords = ['premium', 'luxury', 'deluxe', 'professional', 'elite', 'supreme', 'ultra']
        budget_keywords = ['budget', 'value', 'basic', 'economy', 'affordable', 'discount']
        quality_keywords = ['organic', 'natural', 'fresh', 'pure', 'authentic']
        
        features['has_premium_words'] = [sum(1 for kw in premium_keywords if kw in str(text).lower()) for text in texts]
        features['has_budget_words'] = [sum(1 for kw in budget_keywords if kw in str(text).lower()) for text in texts]
        features['has_quality_words'] = [sum(1 for kw in quality_keywords if kw in str(text).lower()) for text in texts]

        features['capital_word_count'] = [len([w for w in str(text).split() if w and w[0].isupper()]) for text in texts]
        features['number_count'] = [len(re.findall(r'\d+', str(text))) for text in texts]
        features['number_density'] = [len(re.findall(r'\d+', str(text))) / max(len(str(text).split()), 1) for text in texts]
        
        features['has_weight'] = [1 if re.search(r'\d+\s*(oz|lb|kg|g|gram)', str(text).lower()) else 0 for text in texts]
        features['has_volume'] = [1 if re.search(r'\d+\s*(ml|l|liter|fl oz|gallon)', str(text).lower()) else 0 for text in texts]
        features['special_char_count'] = [len(re.findall(r'[^a-zA-Z0-9\s]', str(text))) for text in texts]

        return pd.DataFrame(features)

    def fit_transform(self, texts):
        handcrafted = self.extract_handcrafted_features(texts)

        if self.use_embeddings:
            embeddings = self.text_model.encode([str(text) for text in texts], show_progress_bar=True, batch_size=64)
            embeddings_df = pd.DataFrame(embeddings, columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])])
            return pd.concat([handcrafted, embeddings_df], axis=1)
        else:
            tfidf_features = self.tfidf.fit_transform([str(text) for text in texts])
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
            return pd.concat([handcrafted, tfidf_df], axis=1)

    def transform(self, texts):
        handcrafted = self.extract_handcrafted_features(texts)

        if self.use_embeddings:
            embeddings = self.text_model.encode([str(text) for text in texts], show_progress_bar=True, batch_size=64)
            embeddings_df = pd.DataFrame(embeddings, columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])])
            return pd.concat([handcrafted, embeddings_df], axis=1)
        else:
            tfidf_features = self.tfidf.transform([str(text) for text in texts])
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
            return pd.concat([handcrafted, tfidf_df], axis=1)


class ImageFeatureExtractor:
    """Extract image features"""

    def __init__(self, image_dir='images'):
        print("🖼️  Loading ResNet50...")
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"   ✅ Model on {self.device}")

        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def download_images_batch(self, df, dataset_type='train'):
        print(f"\n📥 Checking {dataset_type} images...")
        output_dir = os.path.join(self.image_dir, dataset_type)
        os.makedirs(output_dir, exist_ok=True)

        image_links = df['image_link'].tolist()
        existing_images = set()
        if os.path.exists(output_dir):
            existing_images = {f for f in os.listdir(output_dir) if f.endswith('.jpg')}
        
        if len(existing_images) >= len(image_links):
            print(f"   ✅ All images present")
            return
        
        print(f"   🔄 Downloading {len(image_links) - len(existing_images)} images...")
        try:
            download_images(image_links, output_dir)
            print(f"   ✅ Download complete")
        except Exception as e:
            print(f"   ⚠️  Download error: {e}")

    def load_image_from_disk(self, image_path):
        try:
            if os.path.exists(image_path):
                return Image.open(image_path).convert('RGB')
            return None
        except:
            return None

    def get_image_path(self, dataset_type, row_index):
        return os.path.join(self.image_dir, dataset_type, f"{row_index}.jpg")

    def extract_features(self, df, dataset_type='train', cache_file=None, force_recompute=False, checkpoint_every=500):
        if cache_file and os.path.exists(cache_file) and not force_recompute:
            print(f"   ✅ Loading cached features")
            return pd.read_pickle(cache_file)

        checkpoint_file = cache_file.replace('.pkl', '_checkpoint.pkl') if cache_file else None
        processed_indices = set()
        checkpoint_features = {}
        start_idx = 0
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            try:
                checkpoint_df = pd.read_pickle(checkpoint_file)
                checkpoint_features = {idx: row.values for idx, row in checkpoint_df.iterrows()}
                processed_indices = set(checkpoint_features.keys())
                start_idx = len(processed_indices)
                print(f"   📂 Resuming from {start_idx}/{len(df)}")
            except:
                pass

        self.download_images_batch(df, dataset_type)

        features_list = [None] * len(df)
        failed_count = 0
        df_reset = df.reset_index(drop=True)
        
        for idx in tqdm(range(len(df_reset)), desc="   Processing", initial=start_idx):
            if idx in processed_indices:
                features_list[idx] = checkpoint_features[idx]
                continue
            
            try:
                image_path = self.get_image_path(dataset_type, idx)
                img = self.load_image_from_disk(image_path)

                if img is None:
                    features = np.zeros(2048)
                    failed_count += 1
                else:
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        features = self.model(img_tensor).squeeze().cpu().numpy()

                features_list[idx] = features
                
                if checkpoint_file and (idx + 1) % checkpoint_every == 0:
                    temp_df = pd.DataFrame([f for f in features_list if f is not None], columns=[f'img_feat_{i}' for i in range(2048)])
                    temp_df.to_pickle(checkpoint_file)

            except:
                features_list[idx] = np.zeros(2048)
                failed_count += 1

        print(f"\n   {'⚠️' if failed_count > 0 else '✅'} Failed: {failed_count}/{len(df)}")
        image_df = pd.DataFrame(features_list, columns=[f'img_feat_{i}' for i in range(2048)])

        if cache_file:
            os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
            image_df.to_pickle(cache_file)
            if checkpoint_file and os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)

        return image_df


class PricePredictionModel:
    """Main prediction model"""

    def __init__(self, use_images=True, use_embeddings=True, fast_mode=False):
        self.use_images = use_images
        self.fast_mode = fast_mode
        self.text_extractor = TextFeatureExtractor(use_embeddings=use_embeddings)

        if use_images:
            self.image_extractor = ImageFeatureExtractor()

        self.scaler = RobustScaler()
        self.models = None

    def _build_models(self, fast_mode=False):
        """Build ensemble - FIXED: Simpler for faster CV"""
        if fast_mode:
            print("   🔧 FAST MODE (single LightGBM)")
            # SINGLE MODEL - much faster for CV!
            return LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.08,
                num_leaves=50,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,  # Use all cores
                verbose=-1
            )
        else:
            print("   🔧 FULL MODE (4-model ensemble)")
            base_models = [
                ('xgb', XGBRegressor(
                    n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.85,
                    colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=2.0,
                    random_state=42, tree_method='hist', n_jobs=-1
                )),
                ('lgb', LGBMRegressor(
                    n_estimators=500, max_depth=8, learning_rate=0.03, num_leaves=60,
                    subsample=0.85, colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=2.0,
                    random_state=42, n_jobs=-1, verbose=-1
                )),
                ('cat', CatBoostRegressor(
                    iterations=500, depth=8, learning_rate=0.03,
                    l2_leaf_reg=3.0, random_seed=42, verbose=False
                )),
                ('gbm', GradientBoostingRegressor(
                    n_estimators=300, max_depth=7, learning_rate=0.05,
                    subsample=0.85, random_state=42
                ))
            ]
            meta_model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
            
            return StackingRegressor(
                estimators=base_models,
                final_estimator=meta_model,
                cv=3,  # Reduced from 5 for speed
                n_jobs=-1
            )

    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE"""
        return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

    def fit(self, train_df, validate=True):
        """FIXED: Training with REAL fold-by-fold progress"""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)

        # Extract features
        print("\n[1/4] 📝 Text features...")
        text_features = self.text_extractor.fit_transform(train_df['catalog_content'].values)

        if self.use_images:
            print("\n[2/4] 🖼️  Image features...")
            image_features = self.image_extractor.extract_features(
                train_df, dataset_type='train',
                cache_file='cache/train_image_features.pkl',
                checkpoint_every=500
            )
            X = pd.concat([text_features, image_features.set_index(text_features.index)], axis=1)
        else:
            X = text_features

        print(f"\n   📊 Total features: {X.shape[1]}")

        y = train_df['price'].values
        y_log = np.log1p(y)

        print(f"\n[3/4] ⚖️  Scaling...")
        X_scaled = self.scaler.fit_transform(X)

        if validate:
            print(f"\n[3.5/4] 🔄 Cross-Validation (5-Fold)")
            print("="*60)
            
            # FIXED: Manual K-Fold with REAL progress
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_model = self._build_models(fast_mode=True)  # Use fast single model
            
            fold_scores_mae = []
            fold_scores_smape = []
            
            for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
                fold_start = time.time()
                print(f"\n📍 Fold {fold_num}/5:")
                
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = y_log[train_idx], y_log[val_idx]
                
                # Train
                print(f"   🏋️  Training...", end='', flush=True)
                cv_model.fit(X_train_fold, y_train_fold)
                print(" ✅")
                
                # Predict
                print(f"   🎯 Predicting...", end='', flush=True)
                y_pred_log = cv_model.predict(X_val_fold)
                y_pred = np.expm1(y_pred_log)
                y_val_real = np.expm1(y_val_fold)
                print(" ✅")
                
                # Metrics
                mae = mean_absolute_error(y_val_real, y_pred)
                smape = self.calculate_smape(y_val_real, y_pred)
                
                fold_scores_mae.append(mae)
                fold_scores_smape.append(smape)
                
                fold_time = time.time() - fold_start
                print(f"   📊 MAE: ${mae:.2f} | SMAPE: {smape:.2f}% | Time: {fold_time:.0f}s")
            
            # Summary
            print("\n" + "="*60)
            print("📊 CROSS-VALIDATION RESULTS:")
            print("="*60)
            print(f"SMAPE: {np.mean(fold_scores_smape):.2f}% (±{np.std(fold_scores_smape):.2f}%)")
            print(f"MAE:   ${np.mean(fold_scores_mae):.2f} (±${np.std(fold_scores_mae):.2f})")
            print(f"Range: {np.min(fold_scores_smape):.2f}% - {np.max(fold_scores_smape):.2f}%")
            print("="*60)

        # Train final model
        print(f"\n[4/4] 🏋️  Training final model...")
        print(f"   ⏱️  Expected: {'3-5' if self.fast_mode else '15-30'} min")
        
        train_start = time.time()
        self.models = self._build_models(fast_mode=self.fast_mode)
        print(f"   🔄 Fitting...")
        self.models.fit(X_scaled, y_log)
        train_time = (time.time() - train_start) / 60
        
        print(f"\n   ✅ Training done in {train_time:.1f}min")

        # Training metrics
        train_pred = np.expm1(self.models.predict(X_scaled))
        train_smape = self.calculate_smape(y, train_pred)
        print(f"   📊 Training SMAPE: {train_smape:.2f}%")

        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)

    def predict(self, test_df):
        """Make predictions"""
        print("\n" + "="*60)
        print("🚀 PREDICTION")
        print("="*60)

        print("\n📝 Text features...")
        text_features = self.text_extractor.transform(test_df['catalog_content'].values)

        if self.use_images:
            print("🖼️  Image features...")
            image_features = self.image_extractor.extract_features(
                test_df, dataset_type='test',
                cache_file='cache/test_image_features.pkl',
                checkpoint_every=500
            )
            X = pd.concat([text_features, image_features.set_index(text_features.index)], axis=1)
        else:
            X = text_features

        print("🎯 Predicting...")
        X_scaled = self.scaler.transform(X)
        predictions = np.expm1(self.models.predict(X_scaled))
        predictions = np.maximum(predictions, 0.01)

        print("✅ Done!")
        return predictions

    def save(self, filepath='models/model.pkl'):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Saved: {filepath}")

    @staticmethod
    def load(filepath='models/model.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def main():
    # Configuration
    USE_IMAGES = False  # Set True if images work
    USE_EMBEDDINGS = True
    VALIDATE = True
    FAST_MODE = False
    
    TRAIN_PATH = 'Data/student_resource/dataset/train.csv'
    TEST_PATH = 'Data/student_resource/dataset/test.csv'
    OUTPUT_PATH = 'submissions/test_out.csv'
    MODEL_PATH = 'models/model_fixed.pkl'

    os.makedirs('cache', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)

    print("\n" + "="*60)
    print("SMART PRODUCT PRICING - FIXED")
    print("="*60)
    print(f"Images: {USE_IMAGES} | Embeddings: {USE_EMBEDDINGS}")
    print(f"GPU: {torch.cuda.is_available()} | Fast: {FAST_MODE}")
    
    print("\n📂 Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    start_time = time.time()
    
    model = PricePredictionModel(
        use_images=USE_IMAGES,
        use_embeddings=USE_EMBEDDINGS,
        fast_mode=FAST_MODE
    )
    model.fit(train_df, validate=VALIDATE)
    model.save(MODEL_PATH)

    predictions = model.predict(test_df)

    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })

    total_time = (time.time() - start_time) / 60

    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(submission['price'].describe())
    print(f"\n⏱️  Total: {total_time:.1f}min")

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved: {OUTPUT_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()