
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import warnings
import sys

warnings.filterwarnings('ignore')

def extract_pack_quantity(text):
    """Enhanced pack quantity extraction - critical for accurate pricing."""
    if pd.isna(text):
        return 1
    
    text = str(text).lower()
    pack_quantity = 1
    
    # Priority patterns (more specific first)
    patterns = [
        (r'pack\s*of\s*(\d+)', 1.0),
        (r'(\d+)\s*-?\s*pack', 1.0),
        (r'(\d+)\s*-?\s*count', 1.0),
        (r'(\d+)\s*ct\b', 1.0),
        (r'case\s*of\s*(\d+)', 1.0),
        (r'(\d+)\s*pieces?', 1.0),
        (r'(\d+)\s*units?', 1.0),
        (r'set\s*of\s*(\d+)', 1.0),
        (r'bundle\s*of\s*(\d+)', 1.0),
        (r'(\d+)\s*per\s*(?:pack|box|case)', 1.0),
        # Item Pack Quantity patterns
        (r'ipq[:\s]*(\d+)', 1.0),
        (r'item\s*pack\s*quantity[:\s]*(\d+)', 1.0),
    ]
    
    found_quantities = []
    for pattern, weight in patterns:
        matches = re.findall(pattern, text)
        if matches:
            found_quantities.extend([int(m) * weight for m in matches])
    
    if found_quantities:
        # Use the maximum found quantity (handles multi-pack cases)
        pack_quantity = max(found_quantities)
        # Sanity check - cap at reasonable value
        pack_quantity = min(pack_quantity, 500)
    
    return pack_quantity

def extract_weight_volume(text):
    """Extract and normalize weight and volume measurements."""
    if pd.isna(text):
        return 0, 0
    
    text = str(text).lower()
    
    # Weight extraction (convert to oz)
    weight_oz = 0
    weight_patterns = [
        (r'(\d+\.?\d*)\s*(?:oz|ounce|ounces)\b', 1.0),
        (r'(\d+\.?\d*)\s*(?:lb|lbs|pound|pounds)\b', 16.0),
        (r'(\d+\.?\d*)\s*(?:g|gram|grams)\b', 0.035274),
        (r'(\d+\.?\d*)\s*(?:kg|kilogram|kilograms)\b', 35.274),
        (r'(\d+\.?\d*)\s*(?:mg|milligram|milligrams)\b', 0.000035274),
    ]
    
    for pattern, multiplier in weight_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                weight_oz += float(match) * multiplier
            except:
                pass
    
    # Volume extraction (convert to fl oz)
    volume_floz = 0
    volume_patterns = [
        (r'(\d+\.?\d*)\s*(?:fl\.?\s*oz|fluid\s*ounce|fluid\s*ounces)\b', 1.0),
        (r'(\d+\.?\d*)\s*(?:ml|milliliter|milliliters)\b', 0.033814),
        (r'(\d+\.?\d*)\s*(?:l|liter|liters)(?!\s*b)\b', 33.814),  # Avoid 'lb'
        (r'(\d+\.?\d*)\s*(?:qt|quart|quarts)\b', 32.0),
        (r'(\d+\.?\d*)\s*(?:gal|gallon|gallons)\b', 128.0),
        (r'(\d+\.?\d*)\s*(?:pt|pint|pints)\b', 16.0),
    ]
    
    for pattern, multiplier in volume_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                volume_floz += float(match) * multiplier
            except:
                pass
    
    return weight_oz, volume_floz

def extract_dimensions(text):
    """Extract product dimensions."""
    if pd.isna(text):
        return 0, 0, 0, 0
    
    text = str(text).lower()
    
    # Common dimension patterns
    dimension_patterns = [
        r'(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*(?:inches?|in|cm)',
        r'(\d+\.?\d*)\s*["\']?\s*[x×]\s*(\d+\.?\d*)\s*["\']?\s*[x×]\s*(\d+\.?\d*)',
    ]
    
    dimensions = []
    for pattern in dimension_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                dims = [float(d) for d in match]
                dimensions.append(dims)
            except:
                pass
    
    if dimensions:
        # Take the first valid dimension set
        dims = dimensions[0]
        volume = dims[0] * dims[1] * dims[2]
        max_dim = max(dims)
        return dims[0], dims[1], dims[2], volume
    
    return 0, 0, 0, 0

def extract_advanced_features(catalog_content):
    """Comprehensive feature extraction optimized for pricing prediction."""
    features = []
    
    for text in catalog_content:
        if pd.isna(text):
            text = ""
        text = str(text)
        text_lower = text.lower()
        
        # Basic text metrics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = max(1, len([s for s in text.split('.') if s.strip()]))
        avg_word_length = np.mean([len(w) for w in text.split()]) if text.split() else 0
        
        # Extract numbers for statistical features
        numbers = re.findall(r'\d+\.?\d*', text)
        numbers = [float(n) for n in numbers if n and float(n) > 0]
        
        max_number = max(numbers) if numbers else 0
        min_number = min(numbers) if numbers else 0
        avg_number = np.mean(numbers) if numbers else 0
        std_number = np.std(numbers) if numbers else 0
        number_count = len(numbers)
        
        # Pack quantity (CRITICAL for pricing)
        pack_quantity = extract_pack_quantity(text)
        
        # Weight and volume
        weight_oz, volume_floz = extract_weight_volume(text)
        
        # Dimensions
        dim1, dim2, dim3, dim_volume = extract_dimensions(text)
        
        # Calculate per-unit metrics
        weight_per_unit = weight_oz / pack_quantity if pack_quantity > 0 else weight_oz
        volume_per_unit = volume_floz / pack_quantity if pack_quantity > 0 else volume_floz
        
        # Total content (weight or volume)
        total_content = max(weight_oz, volume_floz)
        content_per_unit = total_content / pack_quantity if pack_quantity > 0 else total_content
        
        # Category detection (expanded and refined)
        category_keywords = {
            'food': ['food', 'snack', 'candy', 'chocolate', 'cookie', 'sauce', 'soup', 
                    'spice', 'seasoning', 'cereal', 'bar', 'nut', 'chip', 'drink'],
            'beverage': ['coffee', 'tea', 'juice', 'soda', 'water', 'drink', 'beverage'],
            'beauty': ['cosmetic', 'makeup', 'lipstick', 'beauty', 'skincare', 'lotion',
                      'cream', 'serum', 'foundation', 'mascara', 'nail', 'perfume'],
            'health': ['vitamin', 'supplement', 'protein', 'nutrition', 'health', 'medical',
                      'medicine', 'pill', 'capsule', 'tablet'],
            'household': ['cleaning', 'detergent', 'spray', 'dish', 'laundry', 'fabric',
                         'bathroom', 'kitchen', 'cleaner', 'wipe', 'paper', 'towel'],
            'personal_care': ['shampoo', 'conditioner', 'soap', 'toothpaste', 'deodorant',
                             'razor', 'brush', 'comb', 'hair'],
            'baby': ['baby', 'infant', 'diaper', 'wipe', 'formula', 'nursery'],
            'pet': ['pet', 'dog', 'cat', 'animal', 'treat', 'toy'],
            'office': ['pen', 'pencil', 'paper', 'notebook', 'office', 'supply', 'marker'],
            'electronics': ['battery', 'cable', 'charger', 'electronic', 'digital']
        }
        
        category_scores = {}
        for category, keywords in category_keywords.items():
            category_scores[category] = sum(1 for word in keywords if word in text_lower)
        
        # Quality/brand indicators
        premium_keywords = [
            'premium', 'gourmet', 'deluxe', 'luxury', 'professional', 'artisan',
            'organic', 'natural', 'pure', 'authentic', 'imported', 'specialty',
            'select', 'choice', 'prime', 'finest', 'craft', 'exclusive', 'ultra'
        ]
        
        budget_keywords = [
            'value', 'basic', 'economy', 'budget', 'generic', 'simple',
            'affordable', 'discount', 'everyday', 'standard', 'regular', 'classic'
        ]
        
        premium_score = sum(1 for word in premium_keywords if word in text_lower)
        budget_score = sum(1 for word in budget_keywords if word in text_lower)
        quality_differential = premium_score - budget_score
        
        # Size descriptors
        size_keywords = {
            'small': ['mini', 'small', 'travel', 'sample', 'trial'],
            'large': ['large', 'xl', 'jumbo', 'giant', 'family', 'bulk', 'super', 'mega']
        }
        
        small_score = sum(1 for word in size_keywords['small'] if word in text_lower)
        large_score = sum(1 for word in size_keywords['large'] if word in text_lower)
        
        # Pricing indicators
        bulk_keywords = ['bulk', 'wholesale', 'case', 'carton', 'pallet']
        variety_keywords = ['variety', 'assorted', 'mixed', 'pack']
        
        bulk_score = sum(1 for word in bulk_keywords if word in text_lower)
        variety_score = sum(1 for word in variety_keywords if word in text_lower)
        
        # Special features
        has_flavor = int(bool(re.search(r'flavor|flavour|taste', text_lower)))
        has_color = int(bool(re.search(r'color|colour|shade', text_lower)))
        has_scent = int(bool(re.search(r'scent|fragrance|aroma', text_lower)))
        has_size_info = int(weight_oz > 0 or volume_floz > 0 or dim_volume > 0)
        
        # Brand detection (presence of common brand patterns)
        has_brand = int(bool(re.search(r'\b[A-Z][a-z]+\'?s?\b', text)))
        
        # Text complexity
        capital_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, len(text))
        special_char_count = len(re.findall(r'[^\w\s]', text))
        
        # Compile features
        feature_vector = [
            # Basic text features (7)
            char_count, word_count, sentence_count, avg_word_length,
            capital_ratio, digit_ratio, special_char_count,
            
            # Number features (6)
            max_number, min_number, avg_number, std_number, number_count,
            np.log1p(max_number),
            
            # Pack quantity features (5) - CRITICAL
            pack_quantity, np.log1p(pack_quantity), 
            int(pack_quantity > 1), int(pack_quantity > 5), int(pack_quantity > 10),
            
            # Weight/volume features (8)
            weight_oz, volume_floz, np.log1p(weight_oz), np.log1p(volume_floz),
            weight_per_unit, volume_per_unit, total_content, content_per_unit,
            
            # Dimension features (4)
            dim1, dim2, dim3, np.log1p(dim_volume),
            
            # Category scores (10)
            *[category_scores.get(cat, 0) for cat in ['food', 'beverage', 'beauty', 
              'health', 'household', 'personal_care', 'baby', 'pet', 'office', 'electronics']],
            
            # Quality indicators (3)
            premium_score, budget_score, quality_differential,
            
            # Size indicators (2)
            small_score, large_score,
            
            # Pricing indicators (2)
            bulk_score, variety_score,
            
            # Special features (5)
            has_flavor, has_color, has_scent, has_size_info, has_brand,
        ]
        
        features.append(feature_vector)
    
    return np.array(features)

def create_text_features(catalog_content, max_features=1500, vectorizer=None):
    """Create TF-IDF features from text."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=5,
            max_df=0.6,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        cleaned_text = [re.sub(r'\s+', ' ', str(text)).strip() if not pd.isna(text) else "" 
                       for text in catalog_content]
        tfidf_features = vectorizer.fit_transform(cleaned_text).toarray()
        return tfidf_features, vectorizer
    else:
        cleaned_text = [re.sub(r'\s+', ' ', str(text)).strip() if not pd.isna(text) else "" 
                       for text in catalog_content]
        tfidf_features = vectorizer.transform(cleaned_text).toarray()
        return tfidf_features

def train_model(train_df, n_samples=20000):
    """Train optimized ensemble model."""
    print(f"Training with {n_samples} samples...")
    
    # Sample data
    if len(train_df) > n_samples:
        train_sample = train_df.sample(n=n_samples, random_state=42)
    else:
        train_sample = train_df
    
    print("Extracting features...")
    engineered_features = extract_advanced_features(train_sample['catalog_content'])
    text_features, vectorizer = create_text_features(train_sample['catalog_content'])
    
    # Combine features
    X = np.hstack([engineered_features, text_features])
    y = train_sample['price'].values
    
    print(f"Feature shape: {X.shape}")
    print(f"Price range: ${y.min():.2f} - ${y.max():.2f}")
    print(f"Price mean: ${y.mean():.2f}, median: ${np.median(y):.2f}")
    
    # Remove extreme outliers (more conservative)
    q01 = np.percentile(y, 1)
    q99 = np.percentile(y, 99)
    valid_idx = (y >= q01) & (y <= q99)
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"After filtering: {len(y)} samples")
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models
    models = {
        'rf': RandomForestRegressor(
            n_estimators=250,
            max_depth=30,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=250,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        ),
        'et': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'ridge': Ridge(alpha=5.0),
        'lasso': Lasso(alpha=1.0, max_iter=2000)
    }
    
    # Train with cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_scores = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_scaled, y)
        
        # Cross-validation
        fold_scores = []
        for train_idx, val_idx in kf.split(X_scaled):
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            pred = model.predict(X_val_fold)
            
            # Calculate SMAPE
            smape = np.mean(np.abs(pred - y_val_fold) / ((np.abs(y_val_fold) + np.abs(pred)) / 2)) * 100
            fold_scores.append(smape)
        
        model_scores[name] = np.mean(fold_scores)
        print(f"{name} CV SMAPE: {model_scores[name]:.2f}%")
    
    # Calculate weights (inverse of SMAPE)
    total_inverse = sum(1/score for score in model_scores.values())
    weights = {name: (1/score)/total_inverse for name, score in model_scores.items()}
    
    print(f"\nEnsemble weights: {weights}")
    
    # Retrain on full data
    print("\nRetraining on full dataset...")
    for name, model in models.items():
        model.fit(X_scaled, y)
    
    return models, scaler, vectorizer, weights

def predict(models, scaler, vectorizer, weights, test_df):
    """Make predictions on test set."""
    print("\nMaking predictions...")
    
    # Extract features
    engineered_features = extract_advanced_features(test_df['catalog_content'])
    text_features = create_text_features(test_df['catalog_content'], vectorizer=vectorizer)
    
    X_test = np.hstack([engineered_features, text_features])
    X_test_scaled = scaler.transform(X_test)
    
    # Ensemble predictions
    ensemble_pred = np.zeros(len(X_test))
    for name, model in models.items():
        pred = model.predict(X_test_scaled)
        ensemble_pred += weights[name] * pred
        print(f"{name}: ${pred.mean():.2f} (weight: {weights[name]:.3f})")
    
    # Post-processing
    predictions = np.clip(ensemble_pred, 0.5, 2000)
    
    print(f"\nFinal predictions:")
    print(f"  Range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"  Mean: ${predictions.mean():.2f}")
    print(f"  Median: ${np.median(predictions):.2f}")
    
    return predictions

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Product Pricing Solution v2")
    print("=" * 60)
    
    # Load data
    print("\nLoading training data...")
    train_df = pd.read_csv('dataset/train.csv')
    print(f"Training samples: {len(train_df)}")
    
    # Train model
    n_samples = 25000 if '--full' in sys.argv else 20000
    models, scaler, vectorizer, weights = train_model(train_df, n_samples=n_samples)
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv('dataset/test.csv')
    print(f"Test samples: {len(test_df)}")
    
    # Predict
    predictions = predict(models, scaler, vectorizer, weights, test_df)
    
    # Save
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    output_file = 'dataset/test_out.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\n✓ Predictions saved to {output_file}")
    print("=" * 60)