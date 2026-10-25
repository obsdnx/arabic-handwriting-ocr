# -*- coding: utf-8 -*-
# ==============================================================================
# Preamble: CBAM Attention Mechanism Configuration Comparison (R vs K Grid)
# ==============================================================================
# Purpose:
#   Compares a baseline CNN against versions enhanced with CBAM (Convolutional
#   Block Attention Module). Performs a grid search over CBAM's channel
#   reduction ratio (R) and spatial kernel size (K) to evaluate their impact.
#
# Metrics Measured:
#   - Test Accuracy & Loss
#   - Total Parameters (Millions) & Approximate FLOPS (M/GFLOPS)
#   - Training Time (Seconds) & Average Inference Time (ms/img)
#   - Epochs Run (due to EarlyStopping)
#
# Setup:
# - Uses consistent data loading/preprocessing via a dedicated function.
# - Trains models with EarlyStopping(restore_best_weights=True) and ReduceLROnPlateau.
# - Reports key metrics in a final summary table.
# - Suppresses plotting and model saving for focused comparison.
# ==============================================================================

# --- Core Imports ---
import os
import random
import time
import traceback
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Keras/TF Imports ---
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Add, BatchNormalization, Input,
    Activation, Dropout, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Reshape, Concatenate, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.backend import clear_session, int_shape
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# --- Scikit-learn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Kagglehub Import ---
try:
    import kagglehub
except ImportError:
    print("Error: kagglehub not found. Please install it (`pip install kagglehub`). Exiting.")
    exit()

# ==============================================================================
# Configuration & Constants
# ==============================================================================

# --- Reproducibility ---
SEED = 42

# --- Dataset & Image ---
DATASET_NAME = "mohamedgreshamahdi/hijja-dataset"
IMG_DIM = 32 # Assumes square images (IMG_DIM x IMG_DIM)

# --- Model & Training ---
EPOCHS = 150             # Max epochs before early stopping
BATCH_SIZE = 64
FIXED_DROPOUT_RATE = 0.25 # Consistent dropout across models
PATIENCE_ES = 15          # Early stopping patience
PATIENCE_LR = 7           # ReduceLROnPlateau patience
LEARNING_RATE = 0.001

# --- CBAM Grid Search Parameters ---
CBAM_R_VALUES = [32, 16, 8, 4] # Channel reduction ratios
CBAM_K_VALUES = [1, 3, 5, 7] # Spatial kernel sizes

# --- Setup ---
print("Setting random seeds and precision policy...")
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
set_global_policy("float32") # Use float32 for consistency
print(f"Random seeds set to {SEED}. Precision policy set to float32.")


# ==============================================================================
# Helper Functions
# ==============================================================================

def format_large_number(number):
    """Formats large numbers (params, FLOPs) into K, M, G, T."""
    if number is None or not isinstance(number, (int, float, np.number)) or np.isnan(number):
        return "N/A"
    if isinstance(number, str): # Handle case where error string might be passed
        return number
    num_abs = abs(number)
    if num_abs < 1e3: return f"{int(number)}"
    if num_abs < 1e6: return f"{number / 1e3:.2f} K"
    if num_abs < 1e9: return f"{number / 1e6:.2f} M"
    if num_abs < 1e12: return f"{number / 1e9:.2f} G"
    return f"{number / 1e12:.2f} T"

def get_flops(model):
    """
    Estimates the FLOPs (Floating Point Operations) of a Keras model.
    Relies on internal TensorFlow v1 compatibility profiling tools.
    May not work in all TF versions or for all model types.
    """
    if convert_variables_to_constants_v2 is None:
        print("FLOPs calculation skipped (converter unavailable).")
        return None

    print("Estimating FLOPs...")
    total_flops = None
    try:
        # Ensure the model is built and get input spec
        if not model.inputs:
             raise ValueError("Model inputs are not defined. Build or call the model first.")
        input_shape = model.inputs[0].shape[1:] # Exclude batch dim

        # Handle potential None dimensions (often batch or sequence length)
        concrete_shape = [1 if dim is None else dim for dim in input_shape] # Use 1 for None batch dim
        # Fallback if image dimensions are None (less likely here)
        if any(dim is None for dim in concrete_shape[1:]):
            print(f"Warning: Input shape {input_shape} contains None dimensions beyond batch. "
                  f"Using default ({IMG_DIM}, {IMG_DIM}, 1) for FLOPs.")
            concrete_shape = [1, IMG_DIM, IMG_DIM, 1]

        input_spec = tf.TensorSpec(concrete_shape, model.inputs[0].dtype)
        concrete_func = tf.function(model).get_concrete_function(input_spec)
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        # Calculate FLOPs using TensorFlow v1 profiler API
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Check for profiler availability
            profiler_api = getattr(tf.compat.v1, 'profiler', None)
            if profiler_api and hasattr(profiler_api, 'profile'):
                flops_profile = profiler_api.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                total_flops = getattr(flops_profile, 'total_float_ops', None)
                if total_flops is not None:
                    print("FLOPs estimation successful.")
                else:
                    print("Warning: FLOPs profile object lacked 'total_float_ops'.")
            else:
                print("Warning: tf.compat.v1.profiler unavailable. Cannot calculate FLOPs.")

    except Exception as e:
        print(f"WARNING: Could not calculate FLOPs. Error: {e}")
        # traceback.print_exc() # Uncomment for detailed debugging
    return total_flops


# ==============================================================================
# Data Loading and Preprocessing Function
# ==============================================================================

def load_and_preprocess_data(dataset_name, img_dim, validation_split_ratio=0.1, random_seed=SEED):
    """
    Downloads, loads, preprocesses, and splits the specified dataset.

    Args:
        dataset_name (str): Kaggle dataset identifier (e.g., 'user/dataset').
        img_dim (int): Target dimension for reshaping (img_dim x img_dim).
        validation_split_ratio (float): Fraction of training data for validation.
        random_seed (int): Seed for reproducibility in splitting.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, num_classes, input_shape)
               Returns None for all values if an error occurs.
    """
    print(f"\n--- Loading and Preprocessing Dataset: {dataset_name} ---")
    try:
        # --- Download ---
        print("Attempting dataset download via Kagglehub...")
        download_path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {download_path}")

        # --- Locate CSV Files ---
        data_path = download_path
        required_files = ["train_X.csv", "train_Y.csv", "test_X.csv", "test_Y.csv"]
        if not all(os.path.exists(os.path.join(data_path, f)) for f in required_files):
            # Search common subdirectories if not found in root
            base_dataset_name = dataset_name.split('/')[-1]
            potential_paths = [os.path.join(download_path, "data"),
                               os.path.join(download_path, base_dataset_name),
                               download_path]
            found_path = False
            for p_path in potential_paths:
                if os.path.isdir(p_path) and all(os.path.exists(os.path.join(p_path, f)) for f in required_files):
                    data_path = p_path
                    found_path = True
                    print(f"Found CSV files in: {data_path}")
                    break
            if not found_path:
                raise FileNotFoundError(f"Could not locate required CSV files in {download_path} or subdirs.")
        else:
             print(f"Found CSV files directly in: {data_path}")

        # --- Load Data ---
        print("Loading CSV files into DataFrames...")
        train_X_df = pd.read_csv(os.path.join(data_path, "train_X.csv"), header=None)
        train_Y_df = pd.read_csv(os.path.join(data_path, "train_Y.csv"), header=None)
        test_X_df = pd.read_csv(os.path.join(data_path, "test_X.csv"), header=None)
        test_Y_df = pd.read_csv(os.path.join(data_path, "test_Y.csv"), header=None)

        # --- Convert to NumPy & Basic Cleaning ---
        print("Converting to NumPy, handling NaNs, and checking labels...")
        X_train_raw = train_X_df.values
        y_train_raw = train_Y_df.values.flatten()
        X_test_raw = test_X_df.values
        y_test_raw = test_Y_df.values.flatten()

        # Remove rows with any NaN features
        nan_mask_train = ~np.isnan(X_train_raw).any(axis=1)
        nan_mask_test = ~np.isnan(X_test_raw).any(axis=1)
        X_train_cleaned = X_train_raw[nan_mask_train]
        y_train_cleaned = y_train_raw[nan_mask_train]
        X_test_cleaned = X_test_raw[nan_mask_test]
        y_test_cleaned = y_test_raw[nan_mask_test] # Use cleaned test labels

        if X_train_cleaned.shape[0] == 0 or X_test_cleaned.shape[0] == 0:
            raise ValueError("No data remaining after NaN removal.")
        print(f"Data shapes after NaN removal: Train={X_train_cleaned.shape}, Test={X_test_cleaned.shape}")

        # Convert labels to int and determine number of classes (0-based)
        y_train_int = y_train_cleaned.astype(int)
        y_test_int = y_test_cleaned.astype(int)
        min_label = min(np.min(y_train_int), np.min(y_test_int))
        if min_label == 1:
            print("Adjusting labels to be 0-based.")
            y_train_int -= 1
            y_test_int -= 1
        elif min_label < 0:
             raise ValueError("Labels contain negative values.")
        num_classes = max(np.max(y_train_int), np.max(y_test_int)) + 1
        if num_classes <= 1: # Need at least 2 classes for classification
             raise ValueError(f"Detected {num_classes} class(es). Need at least 2.")
        print(f"Number of classes detected: {num_classes}")

        # --- Scale Features ---
        print("Scaling features using StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_cleaned)
        X_test_scaled = scaler.transform(X_test_cleaned)

        # --- Reshape for CNN ---
        print(f"Reshaping data to ({img_dim}, {img_dim}, 1)...")
        expected_features = img_dim * img_dim
        if X_train_scaled.shape[1] != expected_features:
            raise ValueError(f"Feature count mismatch. Expected {expected_features}, got {X_train_scaled.shape[1]}.")
        input_shape = (img_dim, img_dim, 1)
        X_train_reshaped = X_train_scaled.reshape((-1,) + input_shape).astype('float32')
        X_test_reshaped = X_test_scaled.reshape((-1,) + input_shape).astype('float32')
        print(f"Data reshaped: Train={X_train_reshaped.shape}, Test={X_test_reshaped.shape}")

        # --- One-Hot Encode Labels ---
        print("One-hot encoding labels...")
        y_train_cat = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test_int, num_classes=num_classes)

        # --- Split Train/Validation ---
        print(f"Splitting training data (Validation Ratio: {validation_split_ratio})...")
        stratify_targets = y_train_cat if np.min(np.sum(y_train_cat, axis=0)) > 1 else None
        strat_msg = "stratified" if stratify_targets is not None else "non-stratified"
        try:
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_reshaped, y_train_cat,
                test_size=validation_split_ratio,
                random_state=random_seed,
                shuffle=True,
                stratify=stratify_targets
            )
            print(f"Performed {strat_msg} train/validation split.")
        except ValueError as e:
             print(f"Stratify failed ({e}), performing non-stratified split.")
             X_train_final, X_val, y_train_final, y_val = train_test_split(
                 X_train_reshaped, y_train_cat, test_size=validation_split_ratio,
                 random_state=random_seed, shuffle=True, stratify=None)

        print(f"Final shapes: Train={X_train_final.shape}, Val={X_val.shape}, Test={X_test_reshaped.shape}")
        print("--- Data Loading and Preprocessing Complete ---")
        return X_train_final, y_train_final, X_val, y_val, X_test_reshaped, y_test_cat, num_classes, input_shape

    except (FileNotFoundError, ValueError, TypeError, ImportError, Exception) as e:
        print(f"\nERROR during Data Loading/Preprocessing: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None, -1, None

# ==============================================================================
# Attention Mechanism: CBAM (Convolutional Block Attention Module)
# ==============================================================================

def channel_attention_module(input_tensor, ratio, name_prefix="cam"):
    """
    Channel Attention Module (part of CBAM). Focuses on 'what' is informative.

    Args:
        input_tensor: Input tensor.
        ratio (int): Channel reduction ratio for the shared MLP.
        name_prefix (str): Prefix for layer names.

    Returns:
        Tensor: Input tensor scaled by the channel attention map.
    """
    channel_axis = -1
    # Use int_shape which works for both symbolic and eager tensors
    channels = int_shape(input_tensor)[channel_axis]
    if channels is None:
        # Fallback for fully dynamic shape, though less common in typical CNNs after first layer
        channels = tf.shape(input_tensor)[channel_axis]
        print(f"Warning ({name_prefix}): Using dynamic tf.shape for channels.")

    # Calculate reduced filters, ensuring it's at least 1
    reduced_filters = tf.maximum(1, tf.cast(tf.cast(channels, tf.float32) / ratio, tf.int32))
    # Eager execution might resolve channels to an int
    if isinstance(channels, int):
         reduced_filters = max(1, channels // ratio)

    # Shared MLP Layers
    mlp_shared_1 = Dense(reduced_filters, activation='relu', kernel_initializer='he_normal',
                         use_bias=True, name=f"{name_prefix}_mlp1_r{ratio}")
    mlp_shared_2 = Dense(channels, kernel_initializer='he_normal',
                         use_bias=True, name=f"{name_prefix}_mlp2_r{ratio}")

    # Global Average Pooling Path
    avg_pool = GlobalAveragePooling2D(name=f"{name_prefix}_avgpool")(input_tensor)
    avg_pool = Reshape((1, 1, channels), name=f"{name_prefix}_avg_reshape")(avg_pool)
    avg_pool = mlp_shared_1(avg_pool)
    avg_pool = mlp_shared_2(avg_pool)

    # Global Max Pooling Path
    max_pool = GlobalMaxPooling2D(name=f"{name_prefix}_maxpool")(input_tensor)
    max_pool = Reshape((1, 1, channels), name=f"{name_prefix}_max_reshape")(max_pool)
    max_pool = mlp_shared_1(max_pool)
    max_pool = mlp_shared_2(max_pool)

    # Combine and Activate
    attention = Add(name=f"{name_prefix}_add")([avg_pool, max_pool])
    attention = Activation('sigmoid', name=f"{name_prefix}_sigmoid")(attention)

    # Scale input tensor
    return Multiply(name=f"{name_prefix}_scale")([input_tensor, attention])

def spatial_attention_module(input_tensor, kernel_size, name_prefix="sam"):
    """
    Spatial Attention Module (part of CBAM). Focuses on 'where' is informative.

    Args:
        input_tensor: Input tensor (output from Channel Attention).
        kernel_size (int): Kernel size for the convolutional layer.
        name_prefix (str): Prefix for layer names.

    Returns:
        Tensor: Input tensor scaled by the spatial attention map.
    """
    # Pooling across the channel dimension
    avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True),
                      name=f"{name_prefix}_avgpool")(input_tensor)
    max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True),
                      name=f"{name_prefix}_maxpool")(input_tensor)

    # Concatenate pooled features
    concat = Concatenate(axis=-1, name=f"{name_prefix}_concat")([avg_pool, max_pool])

    # Convolution to generate spatial map
    # Note: use_bias=False is common in SAM implementations
    attention = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same',
                       activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
                       name=f"{name_prefix}_conv_k{kernel_size}")(concat)

    # Scale input tensor
    return Multiply(name=f"{name_prefix}_scale")([input_tensor, attention])

def cbam_block(input_tensor, ratio, kernel_size, name_prefix="cbam"):
    """
    Applies the CBAM block: Channel Attention followed by Spatial Attention.

    Args:
        input_tensor: Input tensor for the block.
        ratio (int): Channel reduction ratio for CAM.
        kernel_size (int): Kernel size for SAM convolution.
        name_prefix (str): Prefix for layer names.

    Returns:
        Tensor: Output tensor after applying CBAM.
    """
    # Apply Channel Attention
    x = channel_attention_module(input_tensor, ratio, name_prefix=f"{name_prefix}_cam")
    # Apply Spatial Attention
    x = spatial_attention_module(x, kernel_size, name_prefix=f"{name_prefix}_sam")
    return x

# ==============================================================================
# Model Building Functions
# ==============================================================================

def build_core_cnn(input_shape, num_classes, dropout_rate,
                   attention_module=None, attention_config={}, model_name="Core_CNN"):
    """
    Builds the base 3-stage CNN architecture. Optionally inserts an attention
    module AFTER activation and BEFORE pooling in each stage.

    Args:
        input_shape (tuple): Shape of the input images (H, W, C).
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate applied after pooling in each stage.
        attention_module (function, optional): Function implementing the attention block
                                                (e.g., cbam_block). Defaults to None.
        attention_config (dict, optional): Dictionary of arguments for the attention module.
        model_name (str): Name for the Keras model.

    Returns:
        tf.keras.models.Model: Compiled Keras model.
    """
    inputs = Input(shape=input_shape, name="Input")
    x = inputs
    filters_per_stage = [64, 128, 256]

    for i, filters in enumerate(filters_per_stage):
        stage = i + 1
        # Convolutional Block
        x = Conv2D(filters, (3, 3), padding='same', name=f"conv{stage}")(x)
        x = BatchNormalization(name=f"bn{stage}")(x)
        x = Activation('relu', name=f"relu{stage}")(x)

        # --- Apply Attention Module (if provided) ---
        if attention_module:
            # Dynamically add R and K to name prefix for clarity if they exist
            attn_prefix = f"attn{stage}"
            if 'ratio' in attention_config: attn_prefix += f"_r{attention_config['ratio']}"
            if 'kernel_size' in attention_config: attn_prefix += f"_k{attention_config['kernel_size']}"

            x = attention_module(x, name_prefix=attn_prefix, **attention_config)
            # print(f" > Added {attention_module.__name__} in Stage {stage} with config: {attention_config}") # Optional: log insertion

        # Pooling and Dropout
        x = MaxPooling2D(pool_size=(2, 2), name=f"pool{stage}")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f"dropout{stage}")(x)

    # Classifier Head
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32', name="output")(x)

    # Create and Compile Model
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_baseline_model(input_shape, num_classes, dropout_rate):
    """Builds the baseline CNN without any attention."""
    return build_core_cnn(
        input_shape, num_classes, dropout_rate,
        attention_module=None,
        model_name="Baseline_CNN"
    )

def build_cbam_model(input_shape, num_classes, dropout_rate, cbam_ratio, spatial_kernel):
    """Builds the CNN with CBAM attention blocks."""
    return build_core_cnn(
        input_shape, num_classes, dropout_rate,
        attention_module=cbam_block,
        attention_config={'ratio': cbam_ratio, 'kernel_size': spatial_kernel},
        model_name=f"CBAM_CNN_R{cbam_ratio}_K{spatial_kernel}" # Name reflects R and K
    )

# ==============================================================================
# Training and Evaluation Function
# ==============================================================================

def train_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                          epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Trains a model with early stopping and evaluates it on the test set.

    Args:
        model (tf.keras.models.Model): Compiled Keras model.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        X_test, y_test: Test data and labels.
        epochs (int): Maximum number of epochs.
        batch_size (int): Training batch size.

    Returns:
        dict: Dictionary containing performance metrics ('accuracy', 'loss',
              'training_time_s', 'avg_inference_ms', 'stopped_epoch').
              Values are 'Error' or -1 in case of failure.
    """
    model_name = model.name
    print(f"\n--- Training and Evaluating: {model_name} ---")

    metrics = {
        'accuracy': 'Error', 'loss': 'Error', 'training_time_s': -1.0,
        'avg_inference_ms': -1.0, 'stopped_epoch': -1
    }

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE_ES,
                                   verbose=1, restore_best_weights=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE_LR,
                                  min_lr=1e-7, verbose=1, mode='min')
    callbacks_list = [early_stopping, reduce_lr]

    start_time = time.time()
    training_history = None
    try:
        # --- Training ---
        print(f"Training for up to {epochs} epochs (Batch Size: {batch_size})...")
        training_history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            shuffle=True,
            verbose=0 # Reduce log clutter during grid search
        )
        metrics['training_time_s'] = time.time() - start_time
        print(f"Training finished: {metrics['training_time_s']:.2f}s.")

        # Determine stopped epoch (add 1 because Keras reports 0-based)
        stopped_epoch_val = getattr(early_stopping, 'stopped_epoch', 0)
        metrics['stopped_epoch'] = stopped_epoch_val + 1 if stopped_epoch_val > 0 else epochs

        # --- Evaluation (using weights restored by EarlyStopping) ---
        print(f"Evaluating on test set (best weights restored)...")
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        metrics['loss'] = loss
        metrics['accuracy'] = accuracy
        print(f"  Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # --- Inference Time Measurement ---
        print("Measuring inference time...")
        # Use a potentially larger batch size for inference benchmark
        inference_batch_size = min(batch_size * 4, len(X_test))
        start_inf = time.time()
        _ = model.predict(X_test, batch_size=inference_batch_size, verbose=0)
        end_inf = time.time()
        if len(X_test) > 0:
            metrics['avg_inference_ms'] = ((end_inf - start_inf) / len(X_test)) * 1000
            print(f"  Avg Inference Time: {metrics['avg_inference_ms']:.3f} ms/sample")
        else:
            print("Warning: Test set empty, cannot calculate inference time.")
            metrics['avg_inference_ms'] = 0.0 # Or None

    except Exception as e:
        print(f"\nERROR during training/evaluation for {model_name}: {e}")
        traceback.print_exc()
        # Ensure time is recorded even if error happened after training started
        if start_time and metrics['training_time_s'] < 0:
            metrics['training_time_s'] = time.time() - start_time
        # Try to get stopped epoch if training completed partially
        if training_history and metrics['stopped_epoch'] < 0:
            metrics['stopped_epoch'] = len(training_history.history.get('loss', []))

    print(f"--- Finished Processing {model_name} ---")
    return metrics


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Starting CBAM Attention Configuration Experiment (R vs K Grid) ")
    print(f" Comparing: Baseline vs. CBAM (R={CBAM_R_VALUES}, K={CBAM_K_VALUES}) ")
    print("="*70 + "\n")

    # --- Load Data ---
    data = load_and_preprocess_data(DATASET_NAME, IMG_DIM, validation_split_ratio=0.1, random_seed=SEED)
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, input_shape = data

    if X_train is None: # Check if data loading failed
        print("\n" + "="*70)
        print(" Execution halted: Data loading or preprocessing failed. ")
        print("="*70 + "\n")
    else:
        all_results = {}      # Stores performance metrics {'Experiment Key': metrics_dict}
        complexity_metrics = {} # Stores complexity {'Experiment Key': {'params': N, 'flops': M}}

        # --- Define Experiments ---
        experiment_configs = []
        exp_id_counter = 1

        # 1. Baseline
        experiment_configs.append({
            "id": str(exp_id_counter),
            "type": "Baseline",
            "builder": build_baseline_model,
            "args": {}
        })
        exp_id_counter += 1

        # 2. CBAM Grid
        for k in CBAM_K_VALUES:
            for r in CBAM_R_VALUES:
                experiment_configs.append({
                    "id": str(exp_id_counter),
                    "type": "CBAM",
                    "builder": build_cbam_model,
                    "args": {"cbam_ratio": r, "spatial_kernel": k} # Pass R and K
                })
                exp_id_counter += 1

        print(f"\nDefined {len(experiment_configs)} experiments to run.")

        # --- Run Experiments ---
        for config in experiment_configs:
            exp_key = f"{config['id']}. {config['type']}"
            if config['type'] == "CBAM":
                exp_key += f" (R={config['args']['cbam_ratio']} K={config['args']['spatial_kernel']})"

            print(f"\n{'='*25} Experiment {exp_key} {'='*25}")

            # Reset state for isolation between experiments
            clear_session()
            os.environ['PYTHONHASHSEED'] = str(SEED); random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
            # print(f"Session cleared, seeds reset to {SEED}.") # Optional verbose log

            current_model = None # Ensure model is cleared if build fails
            try:
                # --- Build Model ---
                print("Building model...")
                current_model = config['builder'](
                    input_shape=input_shape,
                    num_classes=num_classes,
                    dropout_rate=FIXED_DROPOUT_RATE,
                    **config['args']
                )
                actual_model_name = current_model.name # Use name assigned by builder
                print(f"Model built: {actual_model_name}")

                # --- Calculate and Store Complexity ---
                num_params = current_model.count_params()
                flops = get_flops(current_model) # Can return None
                complexity_metrics[exp_key] = {
                    'params': num_params,
                    'flops': flops,
                    'actual_name': actual_model_name
                }
                print(f"  Parameters: {num_params:,} ({format_large_number(num_params)})")
                print(f"  Est. FLOPs: {format_large_number(flops)}")

                # --- Optional: Print Model Summary ---
                # print("\nModel Summary:")
                # current_model.summary(line_length=100)

                # --- Train and Evaluate ---
                metrics = train_evaluate_model(
                    model=current_model,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test,
                    epochs=EPOCHS, batch_size=BATCH_SIZE
                )
                all_results[exp_key] = metrics

            except Exception as build_train_err:
                print(f"\nCRITICAL ERROR during experiment {exp_key}: {build_train_err}")
                traceback.print_exc()
                # Record error state for this experiment
                all_results[exp_key] = {
                    'accuracy': 'Build/Train Error', 'loss': 'Error', 'training_time_s': 'Error',
                    'avg_inference_ms': 'Error', 'stopped_epoch': -1
                }
                if exp_key not in complexity_metrics:
                    complexity_metrics[exp_key] = {'params': 'Error', 'flops': 'Error',
                                                  'actual_name': f"{config['type']} ({config['args']})"}

            finally:
                # Clean up memory
                if current_model is not None: del current_model
                gc.collect()
                print(f"--- End of Experiment {config['id']} ---")


        # --- Final Comparison Summary Table ---
        print("\n" + "="*40 + " Final Results: CBAM R vs K Configuration Comparison " + "="*40)
        # Define table format strings
        hdr_fmt = "{:<4} {:<40} {:<12} {:<10} {:<10} {:<15} {:<18} {:<12}"
        row_fmt = "{:<4} {:<40} {:<12} {:<10} {:<10} {:<15} {:<18} {:<12}"
        print(hdr_fmt.format("ID", "Model Architecture", "Test Acc", "Params", "FLOPs", "Train Time(s)", "Infer Time(ms)", "Epochs Run"))
        print("-" * 135) # Adjust width to match format string lengths

        if all_results:
            # Sort by experiment ID for consistent order
            sorted_keys = sorted(all_results.keys(), key=lambda k: int(k.split('.')[0]))

            for exp_key in sorted_keys:
                res = all_results[exp_key]
                comp = complexity_metrics.get(exp_key, {}) # Get complexity info
                model_id = exp_key.split('.')[0]
                # Use the actual TF model name derived during build for display
                display_name = comp.get('actual_name', exp_key.split('.', 1)[1].strip())[:40] # Truncate if needed

                # Format metrics safely, handling potential 'Error' strings or None
                acc_str = f"{res['accuracy']:.4f}" if isinstance(res['accuracy'], (float, int)) else str(res['accuracy'])
                params_str = format_large_number(comp.get('params'))
                flops_str = format_large_number(comp.get('flops'))
                train_t_str = f"{res['training_time_s']:.1f}" if isinstance(res['training_time_s'], (float, int)) else str(res['training_time_s'])
                infer_t_str = f"{res['avg_inference_ms']:.2f}" if isinstance(res['avg_inference_ms'], (float, int)) else str(res['avg_inference_ms'])
                epochs_run_str = str(res.get('stopped_epoch', 'N/A'))

                print(row_fmt.format(model_id, display_name, acc_str, params_str, flops_str, train_t_str, infer_t_str, epochs_run_str))
        else:
            print("No results were recorded. Check for errors during the experiments.")

        print("-" * 135) # Footer line
        print("\nNotes:")
        print(f" - R=N: CBAM Channel Attention reduction ratio N.")
        print(f" - K=N: CBAM Spatial Attention kernel size N.")
        print(f" - Base CNN: 3 stages [Conv(3x3)-BN-ReLU-MaxPool(2x2)], GlobalAvgPool, Dense(softmax).")
        print(f" - Attention added after ReLU, before MaxPool in each stage.")
        print(f" - Trained up to {EPOCHS} epochs with EarlyStopping (patience={PATIENCE_ES}), LR reduction (patience={PATIENCE_LR}).")
        print(f" - Fixed Dropout: {FIXED_DROPOUT_RATE}. Batch Size: {BATCH_SIZE}. Seed: {SEED}.")

    print(f"\n--- CBAM R vs K Configuration Comparison Experiment Complete ---")