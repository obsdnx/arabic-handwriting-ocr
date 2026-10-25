# Exact Replica of the Hijja Baseline with Visualisations
# -*- coding: utf-8 -*-

"""
Hijja Dataset - Exact Replica Model Implementation
==================================================

This script implements a faithful replication of the CNN model architecture
described in the Hijja dataset paper ("Hijja: A Handwritten Arabic Characters
Dataset" by Najwa Altwaijry & Isra Al-Turaiki). It follows the original
design choices: three convolutional blocks with Local Response Normalization
(LRN) and two dense layers with high dropout.

The script performs the full workflow:
1.  Setup: Imports, reproducibility (seeds), performance settings.
2.  Data Handling: Downloads the Hijja dataset via KaggleHub, performs robust
    preprocessing including NaN removal, scaling, reshaping, label adjustment
    (1-based to 0-based), one-hot encoding, and train/validation splitting.
3.  Model Definition: Constructs the specified CNN architecture.
4.  Training: Trains the model using appropriate callbacks (Early Stopping,
    Model Checkpoint, Reduce LR on Plateau).
5.  Evaluation: Assesses the best model on the test set, reporting loss,
    accuracy, classification report, confusion matrix (with analysis),
    per-class accuracy, and inference time.
6.  Analysis: Calculates approximate FLOPS and parameter count.
7.  Visualization: Plots training history and the confusion matrix.

Purpose:
--------
To provide a clear, reproducible, and well-documented baseline implementation
for the Hijja dataset, suitable for assessment and further experimentation.

Author: [Your Name/Assessor]
Date: [Date]
"""

# ==============================================================================
# 1. SETUP: IMPORTS, REPRODUCIBILITY, PERFORMANCE
# ==============================================================================

# --- Essential Imports ---
import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Input, Activation, Dropout, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
print(f"Using TensorFlow Version: {tf.__version__}")
print("Using tf.keras.optimizers.Adam")
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.metrics import classification_report # Explicit import for clarity
import traceback

# --- Kagglehub Import ---
try:
    import kagglehub
    print("Kagglehub imported successfully.")
except ImportError:
    print("ERROR: kagglehub not found. Please install it (`pip install kagglehub`)")
    exit(1) # Exit if essential library is missing

# --- Reproducibility Setup ---
print("\nSetting random seeds for reproducibility...")
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# Optional: Configure TensorFlow for deterministic operations (might impact performance)
# tf.config.experimental.enable_op_determinism()
print(f"Random seeds set to {seed}.")

# --- Performance Setup ---
# Use float32 as per standard practice and likely original paper setup
print("Setting Keras precision policy to 'float32'...")
tf.keras.mixed_precision.set_global_policy("float32")

# --- Constants ---
DATASET_NAME = "mohamedgreshamahdi/hijja-dataset" # KaggleHub identifier
IMG_DIM = 32 # Image dimensions (height and width)
NUM_CHANNELS = 1 # Grayscale images
MODEL_SAVE_PATH_REPLICA = 'best_hijja_replica_model.keras' # File for saving best model

# ==============================================================================
# 2. HELPER FUNCTIONS: LRN, FLOPS
# ==============================================================================

def lrn(x):
    """Applies Local Response Normalization (using TF defaults)."""
    # Parameters (alpha, beta, bias, depth_radius) use tf.nn defaults
    return tf.nn.local_response_normalization(x)

def get_flops(model):
    """
    Estimates approximate FLOPS for a Keras model using tf.profiler.
    Note: This provides an estimation and might not be perfectly accurate.
    Relies on tf.compat.v1 API, which might change in future TF versions.
    """
    if not hasattr(model, 'inputs') or not model.inputs:
        print("Warning: Model inputs not defined. Cannot calculate FLOPS.")
        return -1
    if not all(inp.shape and len(inp.shape) > 1 for inp in model.inputs):
         print("Warning: Model input shapes not fully defined. Cannot calculate FLOPS.")
         return -1

    try:
        # Create a concrete function from the model
        input_signature = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype) for inp in model.inputs]
        concrete_func = tf.function(model).get_concrete_function(input_signature)

        # Use TF profiler to estimate FLOPS
        frozen_func = concrete_func.graph.as_graph_def()
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(frozen_func, name='')
            # Use tf.compat.v1 RunMetadata and Profiler for broader compatibility
            # pylint: disable=E1101
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops_data = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            # pylint: enable=E1101

        return flops_data.total_float_ops if flops_data else -1
    except Exception as e:
        print(f"Warning: Could not calculate FLOPS. Error: {e}")
        # print(traceback.format_exc()) # Uncomment for detailed debug info
        return -1

# ==============================================================================
# 3. DATA HANDLING: LOAD, PREPROCESS, SPLIT
# ==============================================================================

print(f"\nStarting Data Loading and Preprocessing for '{DATASET_NAME}'...")
data_loaded = False
# Initialize variables to store processed data
X_train_split, X_val_split, y_train_split, y_val_split = None, None, None, None
X_test_processed = None # Renamed for clarity
y_test_cat = None
num_classes = -1
class_names = None
y_test_int_labels = None # Stores 0-based integer labels for the test set

try:
    # --- 3.1 Download and Locate Dataset Files ---
    print("Attempting dataset download via KaggleHub...")
    download_root_path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset download/cache path: {download_root_path}")
    data_path = download_root_path # Assume files are in root initially
    required_files = ["train_X.csv", "train_Y.csv", "test_X.csv", "test_Y.csv"]

    # Robustly check for files (root or common subdirectories)
    if not all(os.path.exists(os.path.join(data_path, f)) for f in required_files):
        print(f"Files not in root ({data_path}). Checking subdirectories ('data/', '{DATASET_NAME.split('/')[-1]}/')...")
        potential_paths = [os.path.join(download_root_path, "data"),
                           os.path.join(download_root_path, DATASET_NAME.split('/')[-1])]
        found_path = False
        for p_path in potential_paths:
            if os.path.isdir(p_path) and all(os.path.exists(os.path.join(p_path, f)) for f in required_files):
                data_path = p_path
                found_path = True
                print(f"Found CSV files in subdirectory: {data_path}")
                break
        if not found_path:
            raise FileNotFoundError(f"Could not locate required CSV files in '{download_root_path}' or common subdirectories.")
    else:
        print(f"Found CSV files directly in: {data_path}")

    # --- 3.2 Load CSV Data ---
    print("\nLoading CSV files using pandas (no header)...")
    train_x_path = os.path.join(data_path, "train_X.csv")
    train_y_path = os.path.join(data_path, "train_Y.csv")
    test_x_path = os.path.join(data_path, "test_X.csv")
    test_y_path = os.path.join(data_path, "test_Y.csv")
    train_X_df = pd.read_csv(train_x_path, header=None)
    train_Y_df = pd.read_csv(train_y_path, header=None)
    test_X_df = pd.read_csv(test_x_path, header=None)
    test_Y_df = pd.read_csv(test_y_path, header=None)
    print("CSV files loaded.")

    # --- 3.3 Convert to NumPy and Initial Checks ---
    print("\nConverting dataframes to NumPy arrays...")
    X_train_raw = train_X_df.values
    y_train_raw = train_Y_df.values.flatten() # Ensure labels are 1D
    X_test_raw = test_X_df.values
    y_test_raw = test_Y_df.values.flatten()
    print(f"Raw shapes: Train X={X_train_raw.shape}, Y={y_train_raw.shape} | Test X={X_test_raw.shape}, Y={y_test_raw.shape}")
    if y_train_raw.size == 0 or y_test_raw.size == 0:
        raise ValueError("Label arrays are empty after loading.")

    # --- 3.4 Process Labels (Adjust to 0-based, Determine Classes) ---
    print("\nProcessing labels: Adjusting to 0-based index and determining classes...")
    try:
        # Convert labels to integers and shift from 1-based to 0-based
        y_train_int_labels = y_train_raw.astype(int) - 1
        y_test_int_labels = y_test_raw.astype(int) - 1 # Keep 0-based ints for metrics
        print("Labels adjusted from assumed 1-based to 0-based indexing.")
    except ValueError:
        raise TypeError("Labels contain non-numeric values.")

    if np.min(y_train_int_labels) < 0 or np.min(y_test_int_labels) < 0:
        print(f"Warning: Negative labels found after adjustment (min_train={np.min(y_train_int_labels)}, min_test={np.min(y_test_int_labels)}). Check original label range.")

    # Determine number of classes
    derived_num_classes = max(np.max(y_train_int_labels), np.max(y_test_int_labels)) + 1
    num_classes = derived_num_classes
    if num_classes <= 0: raise ValueError(f"Invalid number of classes detected: {num_classes}.")

    # Assign class names (using predefined Arabic characters if possible)
    hijja_class_names_predefined = [
        'أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص',
        'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء'
    ]
    if num_classes == len(hijja_class_names_predefined):
        class_names = hijja_class_names_predefined
        print(f"Using predefined Arabic class names for {num_classes} classes.")
    else:
        print(f"Warning: Derived class count ({num_classes}) != predefined count ({len(hijja_class_names_predefined)}). Generating default numeric names.")
        class_names = [str(i) for i in range(num_classes)]
    print(f"Number of classes: {num_classes} (labels 0 to {num_classes-1})")

    # --- 3.5 Clean Data (Handle NaNs) ---
    print("\nCleaning data: Removing rows with NaN values...")
    nan_mask_train = ~np.isnan(X_train_raw).any(axis=1)
    X_train_cleaned = X_train_raw[nan_mask_train]
    y_train_cleaned = y_train_int_labels[nan_mask_train] # Keep 0-based labels aligned

    nan_mask_test = ~np.isnan(X_test_raw).any(axis=1)
    X_test_cleaned = X_test_raw[nan_mask_test]
    y_test_int_labels = y_test_int_labels[nan_mask_test] # Update 0-based test labels as well

    print(f"Shapes after NaN removal: Train X={X_train_cleaned.shape}, Y={y_train_cleaned.shape} | Test X={X_test_cleaned.shape}, Y={y_test_int_labels.shape}")
    if len(X_train_cleaned) != len(y_train_cleaned) or len(X_test_cleaned) != len(y_test_int_labels):
        print("CRITICAL WARNING: Feature/label length mismatch after NaN removal!")
    if X_train_cleaned.shape[0] == 0 or X_test_cleaned.shape[0] == 0:
        raise ValueError("No data left after NaN removal.")

    # --- 3.6 Scale Features ---
    print("\nScaling features using StandardScaler (fit on train, transform train/test)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cleaned)
    X_test_scaled = scaler.transform(X_test_cleaned)
    print("Feature scaling complete.")

    # --- 3.7 Reshape Features for CNN ---
    print(f"\nReshaping features for CNN input ({IMG_DIM}x{IMG_DIM}x{NUM_CHANNELS})...")
    expected_features = IMG_DIM * IMG_DIM
    if X_train_scaled.shape[1] != expected_features:
        raise ValueError(f"Feature count mismatch ({X_train_scaled.shape[1]}) vs expected ({expected_features}). Check IMG_DIM.")

    X_train_processed = X_train_scaled.reshape(-1, IMG_DIM, IMG_DIM, NUM_CHANNELS).astype('float32')
    X_test_processed = X_test_scaled.reshape(-1, IMG_DIM, IMG_DIM, NUM_CHANNELS).astype('float32') # Assign final test features
    print(f"Data reshaped: Train={X_train_processed.shape}, Test={X_test_processed.shape}")

    # --- 3.8 One-Hot Encode Labels ---
    print("\nConverting labels to one-hot categorical format...")
    y_train_cat = tf.keras.utils.to_categorical(y_train_cleaned, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test_int_labels, num_classes=num_classes) # Assign final test labels (one-hot)
    print(f"Labels one-hot encoded: Train={y_train_cat.shape}, Test={y_test_cat.shape}")

    # --- 3.9 Split Training Data into Train/Validation ---
    print(f"\nSplitting training data into Training / Validation sets (90% / 10%)...")
    # Use stratify=y_train_cleaned to maintain class distribution if desired, though None is often sufficient with enough data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_processed, y_train_cat,
        test_size=0.1, random_state=seed, shuffle=True, stratify=None
    )
    print(f"Training set shapes:   X={X_train_split.shape}, y={y_train_split.shape}")
    print(f"Validation set shapes: X={X_val_split.shape}, y={y_val_split.shape}")

    data_loaded = True # Flag successful completion
    print("\nData Loading and Preprocessing finished successfully.")

except Exception as e:
    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR during Data Loading/Preprocessing: {type(e).__name__} - {e}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    traceback.print_exc()
    print("\nExecution cannot proceed without data.")
    data_loaded = False
    # No exit() here, let the main execution block handle `data_loaded == False`


# ==============================================================================
# 4. MODEL DEFINITION
# ==============================================================================

def build_exact_hijja_model(input_shape, num_classes_local):
    """
    Builds the CNN model architecture replicating the Hijja baseline.
    Input -> [Conv->ReLU->Pool->LRN] x 3 -> Flatten -> [Dense->ReLU] x 2 -> Dropout -> Dense(softmax)
    """
    inputs = Input(shape=input_shape, name='input_image')

    # Convolutional Block 1
    x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal', name='conv1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = Lambda(lrn, name='lrn1')(x) # Apply Local Response Normalization

    # Convolutional Block 2
    x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = Lambda(lrn, name='lrn2')(x) # Apply Local Response Normalization

    # Convolutional Block 3
    x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal', name='conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    x = Lambda(lrn, name='lrn3')(x) # Apply Local Response Normalization

    # Flatten and Dense Layers
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal', name='dense_1024')(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense_512')(x)
    x = Dropout(0.8, name='dropout_final')(x) # High dropout rate as specified

    # Output Layer
    outputs = Dense(num_classes_local, activation='softmax', dtype='float32', name='output_softmax')(x) # Use float32 for output stability

    # Create and Compile Model
    model = Model(inputs=inputs, outputs=outputs, name="Hijja_Exact_Replica")
    optimizer = Adam(learning_rate=0.001) # Standard Adam optimizer
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# ==============================================================================
# 5. PLOTTING AND ANALYSIS FUNCTIONS
# ==============================================================================

def plot_confusion_matrix(y_true_int, y_pred_int, model_disp_name, class_names_list, num_classes_plot):
    """Generates and displays/saves an annotated confusion matrix heatmap."""
    print(f"\nGenerating Confusion Matrix for {model_disp_name}...")
    if not class_names_list: # Handle case if class_names is None or empty
        class_names_list = [str(i) for i in range(num_classes_plot)]
        print("Warning: class_names_list not provided or empty, generating default labels.")
    if num_classes_plot <= 0:
         print("Warning: Invalid num_classes_plot. Estimating from data.")
         num_classes_plot = max(np.max(y_true_int), np.max(y_pred_int)) + 1


    try:
        cm = metrics.confusion_matrix(y_true_int, y_pred_int)
        plt.figure(figsize=(max(12, num_classes_plot * 0.45), max(10, num_classes_plot * 0.4))) # Adjusted dynamic size
        sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', # Show counts in cells
                    xticklabels=class_names_list[:num_classes_plot], # Ensure list length matches
                    yticklabels=class_names_list[:num_classes_plot],
                    annot_kws={"size": 6 if num_classes_plot > 20 else 8}) # Adjust font size
        plt.title(f'{model_disp_name} - Confusion Matrix', fontsize=14)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout(pad=1.5)
        try:
            plt.show()
        except Exception as plot_e:
            print(f"Note: Could not display confusion matrix plot directly ({plot_e}). Saving...")
            try:
                save_path = f"{model_disp_name.replace(' ', '_')}_confusion_matrix.png"
                plt.savefig(save_path)
                print(f"Confusion matrix saved to {save_path}")
            except Exception as save_e:
                print(f"ERROR: Could not save confusion matrix plot: {save_e}")
            finally:
                plt.close() # Close the figure regardless
        return cm # Return the matrix for analysis
    except Exception as e:
        print(f"ERROR generating confusion matrix: {e}")
        # traceback.print_exc() # Uncomment for detailed debug info
        return None
    finally:
         plt.close('all') # Close all figures just in case

def analyze_confusion_matrix(cm, class_names_list, model_disp_name, top_n=15):
    """Analyzes the confusion matrix to find the most common misclassifications."""
    print(f"\nAnalyzing Confusion Matrix ({model_disp_name})...")
    if cm is None:
        print("Error: Confusion matrix is None, cannot analyze.")
        return
    if not class_names_list:
        class_names_list = [f"Class_{i}" for i in range(cm.shape[0])]
        print("Warning: class_names_list not provided, using default names.")

    try:
        # Create a copy and zero out the diagonal to focus on errors
        cm_errors = cm.copy()
        np.fill_diagonal(cm_errors, 0)

        # Find the indices of the top N largest error counts
        flat_indices = cm_errors.flatten().argsort()[-top_n:][::-1] # Get indices of largest values

        print(f"Top {top_n} Misclassifications (True Class -> Predicted Class):")
        num_printed = 0
        for idx in flat_indices:
            true_idx, pred_idx = np.unravel_index(idx, cm.shape)
            count = cm[true_idx, pred_idx] # Get count from original CM

            if count > 0: # Only report actual errors
                num_printed += 1
                try:
                    true_name = class_names_list[true_idx] if true_idx < len(class_names_list) else f"Idx_{true_idx}"
                    pred_name = class_names_list[pred_idx] if pred_idx < len(class_names_list) else f"Idx_{pred_idx}"
                    print(f"  {num_printed}. '{true_name}' predicted as '{pred_name}': {count} times")
                except IndexError:
                     # This shouldn't happen if class_names_list is handled correctly above
                     print(f"  {num_printed}. Index {true_idx} predicted as Index {pred_idx}: {count} times (Name index error)")
            if num_printed >= top_n:
                break
    except Exception as e:
        print(f"ERROR analyzing confusion matrix: {e}")
        # traceback.print_exc() # Uncomment for detailed debug info


def plot_history(history_obj, model_disp_name):
    """Plots the training and validation loss and accuracy from the Keras history object."""
    print(f"\nPlotting Training History for {model_disp_name}...")
    if not history_obj or not hasattr(history_obj, 'history') or not history_obj.history:
        print("Warning: Invalid or empty history object provided. Skipping plot.")
        return
    required_keys = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    if not all(key in history_obj.history for key in required_keys):
        print(f"Warning: History object missing required keys ({required_keys}). Available: {list(history_obj.history.keys())}. Skipping plot.")
        return

    try:
        history_dict = history_obj.history
        epochs_ran = range(1, len(history_dict['loss']) + 1)

        plt.figure(figsize=(14, 6))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_ran, history_dict['loss'], 'bo-', label='Training Loss')
        plt.plot(epochs_ran, history_dict['val_loss'], 'ro--', label='Validation Loss')
        plt.title(f'{model_disp_name} - Training & Validation Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_ran, history_dict['accuracy'], 'bo-', label='Training Accuracy')
        plt.plot(epochs_ran, history_dict['val_accuracy'], 'ro--', label='Validation Accuracy')
        plt.title(f'{model_disp_name} - Training & Validation Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        # Ensure y-axis starts near reasonable accuracy values if possible
        min_acc = min(min(history_dict['accuracy']), min(history_dict['val_accuracy']))
        if min_acc > 0.5: plt.ylim(bottom=max(0, min_acc - 0.1))


        plt.tight_layout(pad=1.5)
        try:
            plt.show()
        except Exception as plot_e:
            print(f"Note: Could not display history plot directly ({plot_e}).")
            # Optionally save here if display fails
            # plt.savefig(f"{model_disp_name}_history.png")
            plt.close()
    except Exception as e:
         print(f"ERROR plotting history: {e}")
         # traceback.print_exc() # Uncomment for detailed debug info
    finally:
         plt.close('all') # Ensure all plots are closed

# ==============================================================================
# 6. MAIN EXECUTION BLOCK
# ==============================================================================

if data_loaded: # Proceed only if data handling was successful
    print("\n--- Starting Model Training and Evaluation ---")
    input_shape = (IMG_DIM, IMG_DIM, NUM_CHANNELS)
    model_name_display = "Hijja Exact Replica" # Name for display/logging

    # --- 6.1 Build Model ---
    tf.keras.backend.clear_session() # Ensure clean state
    # Seeds are set globally in section 1
    print(f"\nBuilding model: {model_name_display}")
    model = build_exact_hijja_model(input_shape, num_classes)
    model.summary(line_length=110) # Print model summary

    # --- 6.2 Callbacks ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1,
                                   restore_best_weights=True, mode='min')
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH_REPLICA, save_best_only=True,
                                       monitor='val_accuracy', mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                  min_lr=1e-7, verbose=1, mode='min')
    print(f"\nCallbacks configured: EarlyStopping (p=10), ModelCheckpoint ('{MODEL_SAVE_PATH_REPLICA}'), ReduceLROnPlateau (p=5).")

    # --- 6.3 Train the Model ---
    history = None # Initialize history object
    training_time = -1
    print(f"\nTraining {model_name_display}...")
    try:
        start_train_time = time.time()
        history = model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val_split, y_val_split),
            epochs=100, # Set a reasonable max number of epochs
            batch_size=64, # Common batch size
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            shuffle=True, # Shuffle training data each epoch
            verbose=2 # Show one line per epoch
        )
        end_train_time = time.time()
        training_time = end_train_time - start_train_time
        print(f"\nTraining finished in {training_time:.2f} seconds.")
    except Exception as e:
        print(f"\nERROR during model training: {e}")
        traceback.print_exc() # Show full error for training issues
        # Continue execution if possible to evaluate potentially loaded best model

    # --- 6.4 Load Best Model ---
    # If training was interrupted or didn't improve, this loads the best checkpoint
    best_model = None
    if os.path.exists(MODEL_SAVE_PATH_REPLICA):
        print(f"\nLoading best model weights from checkpoint: {MODEL_SAVE_PATH_REPLICA}")
        try:
            # Need custom_objects for the Lambda LRN layer
            best_model = tf.keras.models.load_model(
                MODEL_SAVE_PATH_REPLICA,
                custom_objects={'lrn': lrn}
            )
            print("Best model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load best model from checkpoint ({e}). Falling back to model state at end of training (if available).")
            if 'model' in locals() and isinstance(model, tf.keras.Model):
                 best_model = model # Use the model object from the end of fit()
            else:
                 print("Error: No valid model state available after failed load.")
    elif history is not None and early_stopping.restore_best_weights:
         print("\nUsing model state restored by EarlyStopping callback (checkpoint file not found or load failed).")
         if 'model' in locals() and isinstance(model, tf.keras.Model):
                best_model = model
         else:
             print("Error: No valid model state available from EarlyStopping.")
    else:
        print("\nWarning: No checkpoint file found and EarlyStopping didn't restore weights (or training failed). Evaluation might use untrained/final model.")
        if 'model' in locals() and isinstance(model, tf.keras.Model):
            best_model = model # Default to whatever 'model' is
        else:
            print("Error: No model available for evaluation.")


    # --- 6.5 Model Analysis (FLOPS, Parameters) ---
    if best_model:
        print("\n--- Model Complexity Analysis ---")
        model_flops = get_flops(best_model)
        if model_flops > 0:
            print(f"Approximate FLOPS: {model_flops:,} ({model_flops / 1e9:.2f} GFLOPS)")
        else:
            print("FLOPS calculation failed or not applicable.")
        print(f"Total Parameters: {best_model.count_params():,}")
    else:
        print("\nSkipping complexity analysis: No valid model loaded.")


    # --- 6.6 Evaluate the Best Model ---
    test_accuracy = -1
    test_loss = -1
    if best_model:
        print(f"\n--- Evaluating Best Model on Test Set ---")
        try:
            eval_start_time = time.time()
            # Use the processed test data (X_test_processed, y_test_cat)
            test_loss, test_accuracy = best_model.evaluate(
                X_test_processed, y_test_cat,
                verbose=0 # Set verbose=0 to avoid progress bar during eval printout
            )
            eval_end_time = time.time()
            print(f"Evaluation completed in {eval_end_time - eval_start_time:.2f} seconds.")
            print(f"  Test Loss:     {test_loss:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        except Exception as e:
            print(f"ERROR during final evaluation: {e}")
            traceback.print_exc() # Show full error for evaluation issues
    else:
        print("\nSkipping evaluation: No valid model loaded.")


    # --- 6.7 Predictions & Detailed Metrics ---
    if best_model and test_accuracy != -1: # Only proceed if model exists and evaluation succeeded
        print("\n--- Predictions and Performance Metrics ---")
        try:
            # --- Inference Time ---
            print("Calculating inference time...")
            inference_start_time = time.time()
            # Predict probabilities for the test set
            y_pred_probs = best_model.predict(X_test_processed, batch_size=128) # Use a larger batch size for prediction
            inference_end_time = time.time()
            total_inference_time = inference_end_time - inference_start_time
            avg_inference_time_per_sample = total_inference_time / len(X_test_processed)
            print(f"Inference Time: {total_inference_time:.4f} sec total ({avg_inference_time_per_sample * 1000:.4f} ms/sample)")

            # Convert probabilities to predicted class indices
            y_pred_int = np.argmax(y_pred_probs, axis=1)

            # --- Classification Report ---
            print("\nClassification Report:")
            # Use the 0-based integer test labels (y_test_int_labels) and predicted classes (y_pred_int)
            report = classification_report(
                y_test_int_labels, y_pred_int,
                target_names=class_names, zero_division=0, digits=4
            )
            print(report)

            # --- Confusion Matrix ---
            # Plot and analyze the confusion matrix
            cm = plot_confusion_matrix(y_test_int_labels, y_pred_int, model_name_display, class_names, num_classes)
            if cm is not None:
                analyze_confusion_matrix(cm, class_names, model_name_display)

                # --- Per-Class Accuracy ---
                print("\nPer-Class Accuracy (Recall from CM):")
                with np.errstate(divide='ignore', invalid='ignore'): # Ignore potential division by zero
                    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
                per_class_accuracy = np.nan_to_num(per_class_accuracy) # Replace NaN with 0

                if class_names and len(class_names) == len(per_class_accuracy):
                    for i, acc in enumerate(per_class_accuracy):
                        print(f"  - Class '{class_names[i]}' ({i}): {acc:.4f}")
                else:
                    print("Warning: Class names mismatch or unavailable for per-class accuracy.")
                    print(per_class_accuracy)

        except Exception as e:
            print(f"ERROR during prediction or reporting: {e}")
            traceback.print_exc() # Show full error here
    elif not best_model:
         print("\nSkipping prediction and reporting: No valid model.")
    else:
         print("\nSkipping prediction and reporting due to evaluation failure.")


    # --- 6.8 Plot Training History ---
    if history: # Check if training ran and produced a history object
        plot_history(history, model_name_display)
    else:
        print("\nSkipping history plot: No history object available (training might have failed).")

else: # data_loaded == False
    print("\nExecution halted: Data loading and preprocessing failed. Cannot train or evaluate model.")

# ==============================================================================
print(f"\n--- Hijja Exact Replica Script Execution Complete ---")
# ==============================================================================