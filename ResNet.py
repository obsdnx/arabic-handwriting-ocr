# -*- coding: utf-8 -*-
# ==============================================================================
# Preamble: Unified ResNet Block, Depth, and SE Attention Comparison
# ==============================================================================
# Purpose:
#   Executes a two-phase investigation into ResNet architectures for the
#   Hijja dataset:
#   1. Initial Comparison: Evaluates standard Post-Activation, Pre-Activation,
#      and Bottleneck residual blocks at representative depths (L7, L13, L19).
#   2. SE Enhancement: Applies Squeeze-and-Excitation (SE) attention with
#      varying reduction ratios (R=4, 8, 16, 32) to the best-performing
#      architecture identified in Phase 1 (Standard ResNet L13).
#
# Setup:
# - Uses consistent data loading/preprocessing.
# - Employs a single flexible ResNet builder.
# - Trains models using EarlyStopping and ReduceLROnPlateau.
# - Reports key metrics (Accuracy, Params, FLOPs, Times, Epochs) for all
#   configurations in a final summary table.
# - Suppresses plotting and detailed reports for focused comparison.
# ==============================================================================

# --- Essential Imports ---
import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Add,
                                     BatchNormalization, Input, Activation, Dropout, Multiply,
                                     GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Reshape, Concatenate, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
print("Using tf.keras.optimizers.Adam (Keras 3)")
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.backend import clear_session, int_shape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import traceback
import gc

# FLOPs Calculation Import
try:
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    print("Using tf.python.framework.convert_to_constants.convert_variables_to_constants_v2 for FLOPs.")
except ImportError:
    print("Warning: Could not import convert_variables_to_constants_v2. FLOPs calculation might fail.")
    convert_variables_to_constants_v2 = None

# --- Kagglehub Import and Setup ---
try:
    import kagglehub
    print("Kagglehub imported successfully.")
except ImportError:
    print("Error: kagglehub not found. Please install it (`pip install kagglehub`). Exiting.")
    exit()

# --- Reproducibility Setup ---
print("Setting random seeds...")
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"Random seeds set to {SEED}.")

# --- Performance Setup ---
print("Setting precision policy to float32...")
set_global_policy("float32")

# --- Constants ---
DATASET_NAME = "mohamedgreshamahdi/hijja-dataset"
IMG_DIM = 32
NUM_CHANNELS = 1
EPOCHS = 100 # Max epochs before early stopping
BATCH_SIZE = 64
FIXED_DROPOUT_RATE = 0.2 # Consistent dropout for ResNet experiments
PATIENCE_ES = 15 # Early stopping patience
PATIENCE_LR = 7  # ReduceLROnPlateau patience
LEARNING_RATE = 0.001
BOTTLENECK_EXPANSION = 4 # For Bottleneck blocks
STAGE_FILTERS = [64, 128, 256] # Base filters for stages 1, 2, 3

# ==============================================================================
# Helper Functions (FLOPs, Formatting)
# ==============================================================================

def format_large_number(number):
    """Formats large numbers (params, FLOPs) into K, M, G, T."""
    # (Using the robust version from previous scripts)
    if number is None or not isinstance(number, (int, float, np.number)) or np.isnan(number): return "N/A"
    if isinstance(number, str): return number # Pass through error strings
    num_abs = abs(number)
    if num_abs < 1e3: return f"{int(number)}"
    if num_abs < 1e6: return f"{number / 1e3:.2f} K"
    if num_abs < 1e9: return f"{number / 1e6:.2f} M"
    if num_abs < 1e12: return f"{number / 1e9:.2f} G"
    return f"{number / 1e12:.2f} T"

def get_flops(model, input_shape_concrete):
    """Estimates the FLOPs of a Keras model given a *concrete* input shape."""
    # (Using the robust version from previous scripts)
    if convert_variables_to_constants_v2 is None:
        print("FLOPs calculation skipped: converter unavailable.")
        return None
    if any(dim is None for dim in input_shape_concrete):
         print(f"ERROR: FLOPs calculation requires a concrete input shape. Got {input_shape_concrete}. Skipping.")
         return None

    print("Estimating FLOPs...")
    total_flops = None
    try:
        if not model.built:
            print("Model not built, building with dummy input...")
            dummy_input = tf.zeros([1] + list(input_shape_concrete), dtype=model.inputs[0].dtype)
            _ = model(dummy_input, training=False)

        input_spec = tf.TensorSpec([1] + list(input_shape_concrete), model.inputs[0].dtype)
        concrete_func = tf.function(model).get_concrete_function(input_spec)
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            profiler_api = getattr(tf.compat.v1, 'profiler', None)
            if profiler_api and hasattr(profiler_api, 'profile'):
                flops_profile = profiler_api.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                total_flops = getattr(flops_profile, 'total_float_ops', None)
                if total_flops is not None: print("FLOPs estimation successful.")
                else: print("Warning: FLOPs profile object lacked 'total_float_ops'.")
            else: print("Warning: tf.compat.v1.profiler unavailable.")

    except Exception as e: print(f"WARNING: Could not calculate FLOPs. Error: {e}")
    return total_flops

# ==============================================================================
# Data Loading and Preprocessing Function
# ==============================================================================

def load_and_preprocess_data(dataset_name, img_dim, validation_split_ratio=0.1, random_seed=SEED):
    """Downloads, loads, preprocesses, and splits the Hijja dataset."""
    # (Using the robust version from previous scripts)
    print(f"\n--- Loading and Preprocessing Dataset: {dataset_name} ---")
    data_loaded_success = False
    # Initialize return values
    X_train_final, y_train_final, X_val, y_val, X_test_final, y_test_final = [None] * 6
    num_classes_final, input_shape_final = -1, None

    try:
        print("Attempting dataset download via Kagglehub...")
        download_path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset download/cache path: {download_path}")
        data_path = download_path
        required_files = ["train_X.csv", "train_Y.csv", "test_X.csv", "test_Y.csv"]
        if not all(os.path.exists(os.path.join(data_path, f)) for f in required_files):
             base_dataset_name = dataset_name.split('/')[-1]
             potential_paths = [os.path.join(download_path, "data"), os.path.join(download_path, base_dataset_name), download_path]
             found_path = False
             for p_path in potential_paths:
                 if os.path.isdir(p_path) and all(os.path.exists(os.path.join(p_path, f)) for f in required_files):
                     data_path = p_path; found_path = True; print(f"Found CSV files in: {data_path}"); break
             if not found_path: raise FileNotFoundError(f"Could not locate required CSV files in {download_path} or common subdirs.")
        else: print(f"Found CSV files directly in: {data_path}")

        print("Loading CSV files...")
        train_X_df = pd.read_csv(os.path.join(data_path, "train_X.csv"), header=None)
        train_Y_df = pd.read_csv(os.path.join(data_path, "train_Y.csv"), header=None)
        test_X_df = pd.read_csv(os.path.join(data_path, "test_X.csv"), header=None)
        test_Y_df = pd.read_csv(os.path.join(data_path, "test_Y.csv"), header=None)

        print("Converting to NumPy, handling NaNs, processing labels...")
        X_train_raw = train_X_df.values; y_train_raw = train_Y_df.values.flatten()
        X_test_raw = test_X_df.values; y_test_raw = test_Y_df.values.flatten()
        nan_mask_train = ~np.isnan(X_train_raw).any(axis=1); nan_mask_test = ~np.isnan(X_test_raw).any(axis=1)
        X_train_cleaned = X_train_raw[nan_mask_train]; y_train_cleaned = y_train_raw[nan_mask_train]
        X_test_cleaned = X_test_raw[nan_mask_test]; y_test_cleaned = y_test_raw[nan_mask_test]
        if X_train_cleaned.shape[0] == 0 or X_test_cleaned.shape[0] == 0: raise ValueError("No data after NaN removal.")

        y_train_int = y_train_cleaned.astype(int); y_test_int = y_test_cleaned.astype(int)
        min_label = min(np.min(y_train_int), np.min(y_test_int))
        if min_label == 1: y_train_int -= 1; y_test_int -= 1; print("Adjusted labels to 0-based.")
        elif min_label < 0: raise ValueError("Labels contain negative values.")
        num_classes_final = max(np.max(y_train_int), np.max(y_test_int)) + 1
        if num_classes_final <= 1: raise ValueError(f"Invalid class count: {num_classes_final}.")
        print(f"Number of classes: {num_classes_final}")

        print("Scaling features (StandardScaler)...")
        scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train_cleaned); X_test_scaled = scaler.transform(X_test_cleaned)

        print(f"Reshaping features to ({img_dim}, {img_dim}, 1)...")
        expected_features = img_dim * img_dim
        if X_train_scaled.shape[1] != expected_features: raise ValueError("Feature count mismatch.")
        input_shape_final = (img_dim, img_dim, 1)
        X_train_reshaped = X_train_scaled.reshape((-1,) + input_shape_final).astype('float32')
        X_test_final = X_test_scaled.reshape((-1,) + input_shape_final).astype('float32') # Final Test X

        print("One-hot encoding labels...")
        y_train_cat = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes_final)
        y_test_final = tf.keras.utils.to_categorical(y_test_int, num_classes=num_classes_final) # Final Test y

        print("Splitting training data into Train/Validation...")
        stratify_targets = y_train_cat if np.min(np.sum(y_train_cat, axis=0)) > 1 else None
        strat_msg = "stratified" if stratify_targets is not None else "non-stratified"
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_reshaped, y_train_cat, test_size=validation_split_ratio,
            random_state=random_seed, shuffle=True, stratify=stratify_targets
        )
        print(f"Performed {strat_msg} split.")
        print(f"Final shapes: Train={X_train_final.shape}, Val={X_val.shape}, Test={X_test_final.shape}")
        data_loaded_success = True
        print("--- Data Loading and Preprocessing Complete ---")

    except Exception as e:
        print(f"\nERROR during Data Loading/Preprocessing: {e}"); traceback.print_exc()
        data_loaded_success = False

    if data_loaded_success:
        return X_train_final, y_train_final, X_val, y_val, X_test_final, y_test_final, num_classes_final, input_shape_final
    else:
        return (None,) * 8 # Return Nones on failure

# ==============================================================================
# Attention Block: Squeeze-and-Excitation (SE)
# ==============================================================================

def squeeze_excite_block(input_tensor, ratio=16, name_prefix="se"):
    """Squeeze-and-Excitation block."""
    # (Using the robust version from previous scripts)
    init = input_tensor
    channel_axis = -1
    filters = int_shape(init)[channel_axis]
    if filters is None: filters = tf.shape(init)[channel_axis] # Handle dynamic shapes
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D(name=f"{name_prefix}_squeeze")(init)
    se = Reshape(se_shape, name=f"{name_prefix}_reshape")(se)

    # Calculate reduced filters safely for static or dynamic shapes
    static_filters = tf.get_static_value(filters)
    if static_filters is not None:
         reduced_filters = max(1, static_filters // ratio)
    else: # Tensor input
         reduced_filters = tf.maximum(1, tf.cast(tf.cast(filters, tf.float32) / ratio, tf.int32))

    se = Dense(reduced_filters, activation='relu', kernel_initializer='he_normal', use_bias=False, name=f"{name_prefix}_excite_relu_r{ratio}")(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False, name=f"{name_prefix}_excite_sigmoid_r{ratio}")(se)
    return Multiply(name=f"{name_prefix}_scale")([init, se])

# ==============================================================================
# Residual Block Definitions (Standard, PreActivation, Bottleneck)
# ==============================================================================
# --- Standard Residual Block (Post-activation) with SE Option ---
def standard_residual_block(x, filters, strides=(1, 1), use_projection=False, name_prefix="res_std",
                            use_se=False, se_ratio=16, **kwargs): # Accept extra args
    """Standard residual block with post-activation and optional SE."""
    shortcut = x
    input_filters = int_shape(shortcut)[-1]
    output_filters = filters # Expansion=1 for standard block

    # Main path
    y = Conv2D(filters, (3, 3), strides=strides, padding='same', name=f"{name_prefix}_conv1", kernel_initializer='he_normal')(x)
    y = BatchNormalization(name=f"{name_prefix}_bn1")(y)
    y = Activation('relu', name=f"{name_prefix}_relu1")(y)
    y = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', name=f"{name_prefix}_conv2", kernel_initializer='he_normal')(y)
    y = BatchNormalization(name=f"{name_prefix}_bn2")(y)

    # Apply SE before addition if enabled
    if use_se:
        y = squeeze_excite_block(y, ratio=se_ratio, name_prefix=f"{name_prefix}_se")

    # Shortcut path
    needs_proj = use_projection or strides != (1, 1) or (input_filters is not None and input_filters != output_filters)
    if needs_proj:
        shortcut = Conv2D(output_filters, (1, 1), strides=strides, padding='same', name=f"{name_prefix}_proj_conv", kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    # Combine and final activation
    y = Add(name=f"{name_prefix}_add")([y, shortcut])
    y = Activation('relu', name=f"{name_prefix}_relu_out")(y)
    return y

# --- Pre-activation Residual Block ---
def preactivation_residual_block(x, filters, strides=(1, 1), use_projection=False, name_prefix="res_pre", **kwargs):
    """Residual block with pre-activation (full pre-activation variant)."""
    shortcut = x
    input_filters = int_shape(shortcut)[-1]
    output_filters = filters

    # First pre-activation -> Conv
    y = BatchNormalization(name=f"{name_prefix}_bn1")(x)
    y = Activation('relu', name=f"{name_prefix}_relu1")(y)
    preact_for_shortcut = y # Store this for potential pooling shortcut
    y = Conv2D(filters, (3, 3), strides=strides, padding='same', name=f"{name_prefix}_conv1", kernel_initializer='he_normal', use_bias=False)(y)

    # Second pre-activation -> Conv
    y = BatchNormalization(name=f"{name_prefix}_bn2")(y)
    y = Activation('relu', name=f"{name_prefix}_relu2")(y)
    y = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', name=f"{name_prefix}_conv2", kernel_initializer='he_normal', use_bias=False)(y)

    # Shortcut path
    needs_proj = use_projection or strides != (1, 1) or (input_filters is not None and input_filters != output_filters)
    if needs_proj:
        # Project original input 'x'
        shortcut = Conv2D(output_filters, (1, 1), strides=strides, padding='same', name=f"{name_prefix}_proj_conv", kernel_initializer='he_normal', use_bias=False)(shortcut)
        # No BN on pre-act projection usually
    elif strides != (1, 1) and input_filters == output_filters:
         # Optional: Use pooling on pre-activated output if only downsampling needed
         # shortcut = MaxPooling2D(1, strides=strides, name=f"{name_prefix}_pool_shortcut")(preact_for_shortcut)
         # Sticking to projecting 'x' for simplicity unless pooling proves better
         shortcut = Conv2D(output_filters, (1, 1), strides=strides, padding='same', name=f"{name_prefix}_proj_conv", kernel_initializer='he_normal', use_bias=False)(shortcut)


    # Combine
    y = Add(name=f"{name_prefix}_add")([y, shortcut])
    # No final activation here
    return y

# --- Bottleneck Residual Block ---
def bottleneck_residual_block(x, filters, strides=(1, 1), use_projection=False, expansion=BOTTLENECK_EXPANSION, name_prefix="res_bottle", **kwargs):
    """Bottleneck residual block with post-activation."""
    shortcut = x
    input_filters = int_shape(shortcut)[-1]
    output_filters = filters * expansion # Bottleneck expands channels

    # Main path: 1x1 reduce -> 3x3 -> 1x1 expand
    y = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=f"{name_prefix}_conv1_reduce", kernel_initializer='he_normal')(x)
    y = BatchNormalization(name=f"{name_prefix}_bn1")(y)
    y = Activation('relu', name=f"{name_prefix}_relu1")(y)
    y = Conv2D(filters, (3, 3), strides=strides, padding='same', name=f"{name_prefix}_conv2_main", kernel_initializer='he_normal')(y)
    y = BatchNormalization(name=f"{name_prefix}_bn2")(y)
    y = Activation('relu', name=f"{name_prefix}_relu2")(y)
    y = Conv2D(output_filters, (1, 1), strides=(1, 1), padding='same', name=f"{name_prefix}_conv3_expand", kernel_initializer='he_normal')(y)
    y = BatchNormalization(name=f"{name_prefix}_bn3")(y)

    # Shortcut path
    needs_proj = use_projection or strides != (1, 1) or (input_filters is not None and input_filters != output_filters)
    if needs_proj:
        shortcut = Conv2D(output_filters, (1, 1), strides=strides, padding='same', name=f"{name_prefix}_proj_conv", kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    # Combine and final activation
    y = Add(name=f"{name_prefix}_add")([y, shortcut])
    y = Activation('relu', name=f"{name_prefix}_relu_out")(y)
    return y

# ==============================================================================
# Configurable ResNet Builder
# ==============================================================================

def build_resnet_configurable(input_shape, num_classes, layers_per_stage, block_function,
                              base_filters=64, dropout_rate=FIXED_DROPOUT_RATE,
                              use_se=False, se_ratio=16, block_args={}, model_name_suffix="ResNet"):
    """
    Builds a configurable ResNet model.

    Args:
        input_shape (tuple): Input shape (H, W, C).
        num_classes (int): Number of output classes.
        layers_per_stage (list): List of block counts per stage (e.g., [1, 1, 1] or [2, 2, 2]).
        block_function (function): The residual block function to use.
        base_filters (int): Number of filters in the initial conv layer.
        dropout_rate (float): Dropout rate after pooling.
        use_se (bool): Whether to use SE blocks (only applies if block_function supports it).
        se_ratio (int): Reduction ratio for SE blocks.
        block_args (dict): Additional arguments for the block function (e.g., expansion).
        model_name_suffix (str): Suffix for the Keras model name.

    Returns:
        tf.keras.models.Model: Compiled Keras model.
    """
    inputs = Input(shape=input_shape, name="Input")

    # Initial Convolution Layer
    x = Conv2D(base_filters, (3, 3), strides=(1, 1), padding='same', name="conv1_initial", kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(name="bn1_initial")(x)
    x = Activation('relu', name="relu1_initial")(x)
    last_filters = base_filters
    total_conv_layers = 1

    expansion = block_args.get('expansion', 1) # Default expansion if not specified

    # --- Residual Stages ---
    num_stages = len(layers_per_stage)
    current_stage_filters = base_filters # Filters for the first stage blocks

    for i in range(num_stages):
        stage_num = i + 1
        num_blocks = layers_per_stage[i]
        # print(f"\nBuilding Stage {stage_num}: {num_blocks} x '{block_function.__name__}', Base Filters={current_stage_filters}, Expansion={expansion}")

        # Determine expected output filters for this stage based on the block
        expected_output_filters = current_stage_filters * expansion

        for j in range(num_blocks):
            block_num = j + 1
            block_name_prefix = f"s{stage_num}_block{block_num}"
            # Apply stride (if needed for downsampling) only on the first block of stages > 1
            # Note: Here we downsample with MaxPooling *after* the stage
            strides_block = (1, 1)
            # Projection needed if it's the first block of stage AND filter count changes
            use_proj_block = (j == 0 and last_filters != expected_output_filters)

            x = block_function(
                x,
                filters=current_stage_filters, # Pass stage base filters
                strides=strides_block,
                use_projection=use_proj_block,
                name_prefix=block_name_prefix,
                use_se=use_se, # Pass SE args down
                se_ratio=se_ratio,
                **block_args # Pass expansion etc.
            )
            last_filters = int_shape(x)[-1] # Update filter count based on block output
            # Count conv layers per block
            convs_in_block = 3 if block_function == bottleneck_residual_block else 2
            total_conv_layers += convs_in_block

        # Pooling after stage (except last)
        if i < num_stages - 1: # Corrected condition for pooling
            x = MaxPooling2D(pool_size=(2, 2), name=f"pool{stage_num}")(x)
            if dropout_rate > 0: x = Dropout(dropout_rate, name=f"dropout{stage_num}")(x)
            # print(f"Stage {stage_num} complete. Shape after pool: {int_shape(x)}")

        # Increase filter count for the next stage
        current_stage_filters *= 2

    # Final Layers
    if block_function == preactivation_residual_block:
        print("Adding final BN-ReLU for Pre-activation model before GAP.")
        x = BatchNormalization(name="final_bn_preact")(x)
        x = Activation('relu', name="final_relu_preact")(x)

    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32', name="output")(x)

    # Create Model Name
    se_part = f"_SE_R{se_ratio}" if use_se else ""
    final_model_name = f"Hijja_{model_name_suffix}{se_part}_L{total_conv_layers}"
    print(f"Model '{final_model_name}' built (~{total_conv_layers} conv layers).")

    model = Model(inputs=inputs, outputs=outputs, name=final_model_name)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ==============================================================================
# Training and Evaluation Function (Streamlined)
# ==============================================================================

def train_and_evaluate_model_streamlined(
    model, model_name,
    X_train, y_train_cat, X_val, y_val_cat,
    X_test, y_test_cat,
    epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED
):
    """ Trains, evaluates, calculates key metrics. Suppresses plotting/saving."""
    # (Using the robust version from previous scripts)
    print(f"\n--- Processing Model: {model_name} ---")
    metrics_dict = {'accuracy': 'Error', 'loss': 'Error', 'training_time_s': 'Error',
                    'avg_inference_ms': 'Error', 'stopped_epoch': 'Error'}
    history_obj = None
    start_time = time.time()

    print(f"Callbacks: ES(p={PATIENCE_ES}), ReduceLR(p={PATIENCE_LR})")
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, verbose=1, restore_best_weights=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE_LR, min_lr=1e-7, verbose=1, mode='min')
    callbacks_list = [early_stopping, reduce_lr]

    try:
        print(f"Training {model_name} for up to {epochs} epochs...")
        history_obj = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, shuffle=True, verbose=0)
        metrics_dict['training_time_s'] = time.time() - start_time
        print(f"Training finished: {metrics_dict['training_time_s']:.2f}s.")

        stopped_epoch_val = getattr(early_stopping, 'stopped_epoch', 0)
        metrics_dict['stopped_epoch'] = stopped_epoch_val + 1 if stopped_epoch_val > 0 else epochs
        if stopped_epoch_val > 0: print(f"Early stopping after epoch {metrics_dict['stopped_epoch']}.")
        else: print(f"Completed {metrics_dict['stopped_epoch']} epochs.")

        print(f"Evaluating '{model_name}' on test set (best weights)...")
        final_loss, final_accuracy = model.evaluate(X_test, y_test_cat, batch_size=batch_size, verbose=0)
        metrics_dict['loss'] = final_loss
        metrics_dict['accuracy'] = final_accuracy
        print(f"  Test Loss: {final_loss:.4f}, Test Accuracy: {final_accuracy:.4f}")

        print("Measuring inference time...")
        inf_start = time.time()
        _ = model.predict(X_test, batch_size=min(batch_size * 4, len(X_test)), verbose=0)
        inf_time = time.time() - inf_start
        metrics_dict['avg_inference_ms'] = (inf_time / len(X_test)) * 1000 if len(X_test) > 0 else 'Error'
        if isinstance(metrics_dict['avg_inference_ms'], float):
            print(f"  Avg Inference Time: {metrics_dict['avg_inference_ms']:.3f} ms/sample")

    except Exception as train_e:
        if isinstance(metrics_dict['training_time_s'], str): # If error happened before fit completed
             metrics_dict['training_time_s'] = time.time() - start_time if start_time else 'Error'
        print(f"\nERROR during training/eval for {model_name}: {train_e}"); traceback.print_exc()
        if isinstance(metrics_dict['stopped_epoch'], str) and history_obj and hasattr(history_obj, 'epoch'):
             metrics_dict['stopped_epoch'] = len(history_obj.epoch) if history_obj.epoch else 'Error'

    finally:
        # Ensure numeric metrics are handled correctly if errors occurred
        for key in ['accuracy', 'loss', 'training_time_s', 'avg_inference_ms']:
             if not isinstance(metrics_dict[key], (float, int)): metrics_dict[key] = 'Error'
        if not isinstance(metrics_dict['stopped_epoch'], int): metrics_dict['stopped_epoch'] = 'Error'

    print(f"--- Finished Processing {model_name} ---")
    return metrics_dict


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Starting Unified ResNet Block, Depth, and SE Comparison Experiment ")
    print(" Phase 1: Comparing Block Types & Depths (L7, L13, L19)")
    print(" Phase 2: Applying SE Ratios (4, 8, 16, 32) to best performer (Std L13)")
    print("="*70 + "\n")

    # --- Load Data ---
    data = load_and_preprocess_data(DATASET_NAME, IMG_DIM, validation_split_ratio=0.1, random_seed=SEED)
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, input_shape = data

    if X_train is None: # Check if data loading failed
        print("\n" + "="*70)
        print(" Execution halted: Data loading or preprocessing failed. ")
        print("="*70 + "\n")
    else:
        # Ensure input_shape is concrete for FLOPs calculation
        concrete_input_shape = tuple(input_shape)

        all_results = {}      # Stores performance metrics {'Experiment Key': metrics_dict}
        complexity_metrics = {} # Stores complexity {'Experiment Key': {'params': N, 'flops': M}}

        # --- Define All Experiment Configurations ---
        experiments = []
        base_filters_std = 64
        dropout_std = FIXED_DROPOUT_RATE

        # Phase 1: Initial Block/Depth Comparison
        phase1_configs = [
            # L7 Variants
            {"id": "C1", "name": "StdRes L7", "layers": [1, 1, 1], "block": standard_residual_block, "se": False, "ratio": None, "b_args": {}},
            {"id": "C2", "name": "PreAct L7", "layers": [1, 1, 1], "block": preactivation_residual_block, "se": False, "ratio": None, "b_args": {}},
            # L13 Variants
            {"id": "D1", "name": "StdRes L13", "layers": [2, 2, 2], "block": standard_residual_block, "se": False, "ratio": None, "b_args": {}},
            {"id": "D2", "name": "PreAct L13", "layers": [2, 2, 2], "block": preactivation_residual_block, "se": False, "ratio": None, "b_args": {}},
            # L19 Variant
            {"id": "D3", "name": "Bottle L19", "layers": [2, 2, 2], "block": bottleneck_residual_block, "se": False, "ratio": None, "b_args": {'expansion': BOTTLENECK_EXPANSION}},
        ]
        experiments.extend(phase1_configs)

        # Phase 2: SE Enhancement on StdRes L13 (Assumed best performer from Phase 1)
        se_ratios_to_test = [4, 8, 16, 32]
        phase2_configs = []
        # Note: The non-SE baseline (D1) is already in phase1_configs
        # We use a different ID prefix 'SE' for clarity in the table
        for r in se_ratios_to_test:
            phase2_configs.append(
                {"id": f"SE{r}", "name": f"StdRes L13 SE R{r}", "layers": [2, 2, 2], "block": standard_residual_block, "se": True, "ratio": r, "b_args": {}}
            )
        experiments.extend(phase2_configs)

        print(f"Defined {len(experiments)} experiments to run.")

        # --- Run Experiments ---
        for config in experiments:
            # Construct a unique key and descriptive name for results storage
            exp_key = f"{config['id']}. {config['name']}"
            print(f"\n{'='*25} Experiment {exp_key} {'='*25}")

            # Reset state
            clear_session(); gc.collect()
            os.environ['PYTHONHASHSEED'] = str(SEED); random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

            current_model = None
            try:
                # --- Build Model ---
                print(f"Building model: {config['name']}...")
                current_model = build_resnet_configurable(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    layers_per_stage=config["layers"],
                    block_function=config["block"],
                    base_filters=base_filters_std,
                    dropout_rate=dropout_std,
                    use_se=config["se"],
                    se_ratio=config["ratio"],
                    block_args=config["b_args"],
                    model_name_suffix=config["name"].replace(" ", "") # Use descriptive suffix
                )
                actual_model_name = current_model.name
                print(f"Model built: {actual_model_name}")

                # --- Calculate Complexity ---
                num_params = current_model.count_params()
                flops = get_flops(current_model, concrete_input_shape) # Pass concrete shape
                complexity_metrics[exp_key] = {
                    'params': num_params, 'flops': flops, 'actual_name': actual_model_name
                }
                print(f"  Parameters: {num_params:,} ({format_large_number(num_params)})")
                print(f"  Est. FLOPs: {format_large_number(flops)}")

                # --- Train and Evaluate ---
                metrics = train_and_evaluate_model_streamlined(
                    model=current_model, model_name=actual_model_name,
                    X_train=X_train, y_train_cat=y_train,
                    X_val=X_val, y_val_cat=y_val,
                    X_test=X_test, y_test_cat=y_test,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED
                )
                all_results[exp_key] = metrics

            except Exception as build_train_err:
                print(f"\nCRITICAL ERROR during experiment {exp_key}: {build_train_err}")
                traceback.print_exc()
                all_results[exp_key] = {'accuracy': 'Error', 'loss': 'Error', 'training_time_s': 'Error',
                                        'avg_inference_ms': 'Error', 'stopped_epoch': 'Error'}
                if exp_key not in complexity_metrics:
                    complexity_metrics[exp_key] = {'params': 'Error', 'flops': 'Error', 'actual_name': config['name']}

            finally:
                if current_model is not None: del current_model
                gc.collect()
                print(f"--- End of Experiment {config['id']} ---")


        # --- Final Comparison Summary Table ---
        print("\n" + "="*40 + " Final Results: Unified ResNet Comparison " + "="*40)
        hdr_fmt = "{:<7} {:<55} {:<12} {:<10} {:<10} {:<15} {:<18} {:<12}"
        row_fmt = "{:<7} {:<55} {:<12} {:<10} {:<10} {:<15} {:<18} {:<12}"
        print(hdr_fmt.format("ID", "Model Architecture", "Test Acc", "Params", "FLOPs", "Train Time(s)", "Infer Time(ms)", "Epochs Run"))
        print("-" * 142) # Adjust width

        if all_results:
            # Sort keys based on the defined ID (C1, C2, D1, D2, D3, SE4, SE8...)
            def sort_key_func(key):
                id_part = key.split('.')[0]
                if id_part.startswith('C'): return (0, int(id_part[1:])) # Phase 1 - L7
                if id_part.startswith('D'): return (1, int(id_part[1:])) # Phase 1 - L13/L19
                if id_part.startswith('SE'): return (2, int(id_part[2:])) # Phase 2 - SE
                return (99, 0) # Fallback
            sorted_keys = sorted(all_results.keys(), key=sort_key_func)

            for exp_key in sorted_keys:
                res = all_results[exp_key]
                comp = complexity_metrics.get(exp_key, {})
                model_id = exp_key.split('.')[0]
                display_name = comp.get('actual_name', exp_key.split('.', 1)[1].strip())[:55] # Use actual TF name

                # Format metrics safely
                acc_str = f"{res['accuracy']:.4f}" if isinstance(res['accuracy'], float) else str(res['accuracy'])
                params_str = format_large_number(comp.get('params'))
                flops_str = format_large_number(comp.get('flops'))
                train_t_str = f"{res['training_time_s']:.1f}" if isinstance(res['training_time_s'], float) else str(res['training_time_s'])
                infer_t_str = f"{res['avg_inference_ms']:.2f}" if isinstance(res['avg_inference_ms'], float) else str(res['avg_inference_ms'])
                epochs_run_str = str(res.get('stopped_epoch', 'N/A'))

                print(row_fmt.format(model_id, display_name, acc_str, params_str, flops_str, train_t_str, infer_t_str, epochs_run_str))
        else:
            print("No results were recorded.")

        print("-" * 142)
        print("\nNotes:")
        print(" - Phase 1 (IDs C1-D3): Compares ResNet block types (Std, PreAct, Bottle) at different depths.")
        print(" - Phase 2 (IDs SE4-SE32): Applies SE Attention (varying ratio R) to StdRes L13 baseline.")
        print(f" - Consistent settings: Dropout={FIXED_DROPOUT_RATE}, Max Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, Seed={SEED}.")

    print(f"\n--- Unified ResNet Experiment Script Finished ---")