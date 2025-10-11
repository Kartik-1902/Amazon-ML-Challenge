import pandas as pd
import numpy as np
import re
import os
import sys
from tqdm import tqdm

# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Model
import lightgbm as lgb

# Image Processing (optional but recommended)
import torch
from torchvision import models, transforms
from PIL import Image
import warnings

# --- Setup ---
# Add src directory to path to import utils
sys.path.append('Data/student_resource/src')
try:
    from utils import download_images
except ImportError:
    print("Could not import download_images from utils.py. Image processing will be skipped.")
    print("Please ensure 'Data/student_resource/src/utils.py' exists.")
    download_images = None

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = 'Data/student_resource/dataset/'
IMAGE_DIR = 'product_images' # Directory to save downloaded images
PROCESS_IMAGES = True # Set to False to skip image processing for a faster run

# Create image directory if it doesn't exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# --- Helper Functions ---

def extract_ipq(text):
    """Extracts Item Pack Quantity (IPQ) from text using regex."""
    text = str(text).lower()
    # Regex to find patterns like "pack of 12", "12 count", "pk/12", etc.
    match = re.search(r'(?:pack of|pk|pack|count|ct|set of)\s*:?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return 1 # Default to 1 if no pack size is found

def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# --- Main Script ---

if __name__ == "__main__":
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    
    # For faster development, you can use a smaller subset
    # train_df = train_df.head(1000)
    # test_df = test_df.head(1000)

    print("Data loaded. Shapes:", train_df.shape, test_df.shape)

    # ===============================================
    # 1. Text Feature Engineering
    # ===============================================
    print("\n--- Starting Text Feature Engineering ---")

    # Combine data for consistent processing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Extract IPQ
    print("Extracting Item Pack Quantity (IPQ)...")
    combined_df['ipq'] = combined_df['catalog_content'].apply(extract_ipq)

    # Clean catalog_content
    print("Cleaning text data...")
    combined_df['cleaned_content'] = combined_df['catalog_content'].apply(clean_text)

    # TF-IDF Vectorization
    print("Applying TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_features = tfidf.fit_transform(combined_df['cleaned_content'])

    # Final text features
    text_features = pd.DataFrame(combined_df[['ipq']])
    
    print("Text feature engineering complete.")

    # ===============================================
    # 2. Image Feature Engineering (Optional)
    # ===============================================
    image_embeddings = None
    if PROCESS_IMAGES and download_images is not None:
        print("\n--- Starting Image Feature Engineering ---")
        
        # Define image paths
        combined_df['image_path'] = combined_df['image_link'].apply(lambda url: os.path.join(IMAGE_DIR, url.split('/')[-1]))

        # --- Download Images ---
        # Note: This can take a very long time for the full dataset!
        print("Downloading images... (This may take a while)")
        # We will download images that don't exist yet
        images_to_download = combined_df[~combined_df['image_path'].apply(os.path.exists)][['image_link', 'image_path']]
        if not images_to_download.empty:
            download_images(images_to_download.rename(columns={'image_path': 'file_path'}))
        else:
            print("All images seem to be downloaded already.")

        # --- Generate Image Embeddings ---
        print("Generating image embeddings with ResNet-50...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load pre-trained ResNet-50
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the final classification layer
        model.to(device)
        model.eval()

        # Image transformations
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def get_embedding(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                img_t = preprocess(img)
                batch_t = torch.unsqueeze(img_t, 0).to(device)
                with torch.no_grad():
                    embedding = model(batch_t)
                return embedding.squeeze().cpu().numpy()
            except Exception as e:
                # print(f"Error processing {image_path}: {e}")
                return np.zeros(2048) # Return a zero vector on error

        # Generate embeddings for all images
        embeddings_list = [get_embedding(p) for p in tqdm(combined_df['image_path'], desc="Processing Images")]
        image_embeddings = np.array(embeddings_list)
        print("Image embeddings generated. Shape:", image_embeddings.shape)

    # ===============================================
    # 3. Combine Features and Prepare for Training
    # ===============================================
    print("\n--- Combining Features and Training Model ---")
    
    from scipy.sparse import hstack, csr_matrix

    # Convert dense text features to sparse matrix
    text_features_sparse = csr_matrix(text_features.values)

    # Combine sparse TF-IDF with other text features
    X_text_combined = hstack([text_features_sparse, tfidf_features])

    # Combine with image features if available
    if image_embeddings is not None:
        image_embeddings_sparse = csr_matrix(image_embeddings)
        X_combined = hstack([X_text_combined, image_embeddings_sparse])
    else:
        X_combined = X_text_combined

    print("Final combined feature matrix shape:", X_combined.shape)

    # Split back into train and test sets
    train_len = len(train_df)
    X_train = X_combined[:train_len]
    X_test = X_combined[train_len:]
    
    # Target variable transformation (log1p)
    y_train = np.log1p(train_df['price'])

    # ===============================================
    # 4. Model Training (LightGBM)
    # ===============================================
    print("Training LightGBM model...")
    lgb_params = {
        'objective': 'regression_l1', # MAE is robust to outliers
        'metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }

    model = lgb.LGBMRegressor(**lgb_params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=True)])

    # ===============================================
    # 5. Prediction and Submission
    # ===============================================
    print("\n--- Generating Predictions ---")
    predictions_log = model.predict(X_test)

    # Inverse transform to get actual price predictions
    predictions = np.expm1(predictions_log)
    
    # Ensure prices are positive
    predictions[predictions < 0] = 0 

    # Create submission file
    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': predictions})
    submission_df.to_csv('test_out.csv', index=False)

    print("\nSubmission file 'test_out.csv' created successfully!")
    print(submission_df.head())