# Code Block: Data Loading and Preprocessing (Simplified based on successful log)

#Data Loading and Preprocessing
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Assuming kagglehub is imported successfully earlier
import kagglehub

# --- Constants in this Block ---
DATASET_NAME = "mohamedgreshamahdi/hijja-dataset"
IMG_DIM = 32
seed = 42 # From the log

# --- Initialize Output Variables ---
X_train_split, X_val_split, y_train_split, y_val_split = None, None, None, None
X_test_scaled_reshaped = None
y_test_cat = None
num_classes = -1
class_names = None
y_test_orig_labels = None # Stores 0-based integer labels for the test set

# ==============================================================================
# --- Data Loading and Preprocessing Logic ---
# ==============================================================================
print(f"\nAcquiring dataset '{DATASET_NAME}'...")

# 1. Download - Assume success and files are directly in the path
print("Attempting dataset download...")
path = kagglehub.dataset_download(DATASET_NAME)
data_path = path # Directly use the returned path
print(f"Using dataset path: {data_path}")

# 2. Load CSV Files - Assume files exist and are readable
print("\nLoading dataset files using pandas (header=None)...")
train_x_path = os.path.join(data_path, "train_X.csv")
train_y_path = os.path.join(data_path, "train_Y.csv")
test_x_path = os.path.join(data_path, "test_X.csv")
test_y_path = os.path.join(data_path, "test_Y.csv")

train_X_df = pd.read_csv(train_x_path, header=None)
train_Y_df = pd.read_csv(train_y_path, header=None)
test_X_df = pd.read_csv(test_x_path, header=None)
test_Y_df = pd.read_csv(test_y_path, header=None)
print("CSV files loaded successfully.")

# 3. Convert to NumPy Arrays
print("\nConverting dataframes to NumPy arrays...")
X_train_raw = train_X_df.values
y_train_raw = train_Y_df.values.flatten()
X_test_raw = test_X_df.values
y_test_raw = test_Y_df.values.flatten()

# 4. Label Processing - Assume 1-based integer labels, no NaNs
print("\nAdjusting labels (assuming 1-based)...")
y_train_labels = y_train_raw.astype(int) - 1 # Adjusted labels (0-based)
y_test_orig_labels = y_test_raw.astype(int) - 1 # Adjusted labels (0-based) for metrics
print("Labels adjusted to 0-based indexing.")

# Determine number of classes - Assume consistency
num_classes = max(np.max(y_train_labels), np.max(y_test_orig_labels)) + 1

# Assign class names - Assume match with predefined
hijja_class_names_predefined = [
    'أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص',
    'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء'
]
class_names = hijja_class_names_predefined # Directly assign based on successful log
print(f"Using predefined Arabic class names for {num_classes} classes.")
print(f"Number of classes determined: {num_classes} (labels 0 to {num_classes-1})")

# 5. Feature Scaling - Apply directly as no NaN cleaning needed
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw) # Use raw data
X_test_scaled = scaler.transform(X_test_raw)     # Use raw data

# 6. Reshape Data for CNN - Assume feature count matches IMG_DIM * IMG_DIM
print(f"\nReshaping data for CNN input ({IMG_DIM}, {IMG_DIM}, 1)...")
X_train_scaled_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], IMG_DIM, IMG_DIM, 1).astype('float32')
X_test_scaled_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], IMG_DIM, IMG_DIM, 1).astype('float32')
print(f"Data reshaped to: Train={X_train_scaled_reshaped.shape}, Test={X_test_scaled_reshaped.shape}")

# 7. One-Hot Encode Labels
print("\nConverting labels to one-hot categorical format...")
y_train_cat = tf.keras.utils.to_categorical(y_train_labels, num_classes=num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test_orig_labels, num_classes=num_classes)
print(f"Labels converted. Shape: Train={y_train_cat.shape}, Test={y_test_cat.shape}")

# 8. Train/Validation Split
print(f"\nSplitting training data into Training/Validation (test_size=0.1, random_state={seed}, shuffle=True)...")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled_reshaped,
    y_train_cat,
    test_size=0.1,
    random_state=seed,
    shuffle=True,
    stratify=None # Default behavior
)
print(f"Training split shapes: X={X_train_split.shape}, y={y_train_split.shape}")
print(f"Validation split shapes: X={X_val_split.shape}, y={y_val_split.shape}")

# --- Set Data Loaded Flag ---
data_loaded = True # Indicate successful completion of this block
print("\nSimplified data loading and preprocessing complete.")

# ==============================================================================
# --- End of Simplified Data Loading and Preprocessing Logic ---
# ==============================================================================

# --- Post-Block Check (Optional but good practice) ---
if data_loaded:
    print("\nOutput variables are ready for model training and evaluation.")
else:
    # This part should theoretically not be reached with the simplified code
    print("\nData loading failed.")