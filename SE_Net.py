# -*- coding: utf-8 -*-
# ==============================================================================
# Preamble: SE Attention Mechanism Placement Comparison (Refined & Simplified)
# ==============================================================================
"""
SE Attention Placement Comparison on Hijja Dataset (Simplified)
==============================================================

Purpose:
--------
Compares a baseline CNN against versions enhanced with Squeeze-and-Excitation (SE)
attention, focusing on the impact of SE block placement ('before_activation' vs.
'after_activation') and different reduction ratios (R=32, 16, 8, 4).

Workflow:
---------
1.  **Setup:** Imports, reproducibility seeds, TF configuration.
2.  **Simplified Data Handling:** Uses a pre-defined, simplified data loading
    block (ASSUMES data is clean, labels are 1-based, files are accessible).
3.  **Model Building:** Defines baseline and SE-enhanced CNNs.
4.  **Experiment Execution:** Runs each defined configuration, calculates complexity,
    trains with EarlyStopping, evaluates, and collects key metrics.
5.  **Summary Report:** Presents a final comparative table.

Streamlining Notes:
-------------------
- Assumes the simplified data loading block runs successfully beforehand.
- Visual plotting, detailed reports, and model saving are disabled.
- Focuses on the final summary table metrics.
"""
# ==============================================================================
# 1. SETUP: IMPORTS, CONFIGURATION, CONSTANTS
# ==============================================================================

# --- Essential Imports ---
import os
import random
import time
import numpy as np
import pandas as pd # Keep for simplified data loading block
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     BatchNormalization, Input, Activation, Dropout, Multiply,
                                     GlobalAveragePooling2D, Reshape) # Removed Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Removed plotting/detailed metrics imports
from sklearn.model_selection import train_test_split # Keep for simplified data block
from sklearn.preprocessing import StandardScaler # Keep for simplified data block
import gc # Garbage Collector

print(f"Using TensorFlow Version: {tf.__version__}")
print("Using tf.keras.optimizers.Adam")

# --- FLOPs Calculation Utility ---
try:
    # Used for FLOPs estimation, relies on TF internals
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    print("FLOPs estimation utility found.")
except ImportError:
    print("Warning: convert_variables_to_constants_v2 not found. FLOPs calculation skipped.")
    convert_variables_to_constants_v2 = None

# --- Kagglehub Import (Needed for Simplified Data Block) ---
try:
    import kagglehub
    print("Kagglehub imported successfully.")
except ImportError:
    print("ERROR: kagglehub not found. Install or check environment. Exiting.")
    exit(1)

# --- Reproducibility ---
print("\nSetting random seeds...")
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
print(f"Random seeds set to {seed}.")

# --- Performance Configuration ---
print("Setting Keras precision policy to 'float32'...")
tf.keras.mixed_precision.set_global_policy("float32")

# --- Constants ---
DATASET_NAME = "mohamedgreshamahdi/hijja-dataset" # Keep for data block
IMG_DIM = 32
NUM_CHANNELS = 1
EPOCHS = 150           # Max epochs
BATCH_SIZE = 64
FIXED_DROPOUT_RATE = 0.25 # Consistent dropout
PATIENCE_ES = 15        # Early stopping patience
PATIENCE_LR = 7         # ReduceLROnPlateau patience

# ==============================================================================
# 2. DATA HANDLING (Simplified Block Integration)
# ==============================================================================

# ------------------------------------------------------------------------------
# PREAMBLE for Simplified Data Loading: ASSUMPTIONS MADE
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

# --- Start of  Data Loading Block ---

# Data Loading and Preprocessing

# --- Initialize Output Variables ---
# These will be populated by the data loading block
X_train_split, X_val_split, y_train_split, y_val_split = None, None, None, None
X_test_processed = None # Renamed for clarity post-processing
y_test_cat = None
num_classes = -1
# class_names = None # Define later if needed for summary only
y_test_int_labels = None # Renamed from y_test_orig_labels/cleaned

# --- Data Loading and Preprocessing Logic ---
print(f"\nAcquiring dataset '{DATASET_NAME}' (Simplified Mode)...")
data_loaded = False
try:
    # 1. Download
    print("Attempting dataset download...")
    path = kagglehub.dataset_download(DATASET_NAME)
    data_path = path
    print(f"Using dataset path: {data_path}")

    # 2. Load CSV Files
    print("\nLoading dataset files (pandas)...")
    read_opts = {'header': None}
    train_X_df = pd.read_csv(os.path.join(data_path, "train_X.csv"), **read_opts)
    train_Y_df = pd.read_csv(os.path.join(data_path, "train_Y.csv"), **read_opts)
    test_X_df  = pd.read_csv(os.path.join(data_path, "test_X.csv"), **read_opts)
    test_Y_df  = pd.read_csv(os.path.join(data_path, "test_Y.csv"), **read_opts)
    print("CSV files loaded.")

    # 3. Convert to NumPy
    print("\nConverting to NumPy arrays...")
    X_train_raw = train_X_df.values; y_train_raw = train_Y_df.values.flatten()
    X_test_raw  = test_X_df.values;  y_test_raw  = test_Y_df.values.flatten()

    # 4. Process Labels (Assume 1-based)
    print("\nProcessing labels (adjusting to 0-based)...")
    y_train_int_labels = y_train_raw.astype(int) - 1
    y_test_int_labels = y_test_raw.astype(int) - 1 # Keep 0-based for potential later use
    print("Labels adjusted.")

    # Determine number of classes
    num_classes = max(np.max(y_train_int_labels), np.max(y_test_int_labels)) + 1
    print(f"Number of classes determined: {num_classes}")
    # Define class names here if needed, or defer. For summary, only num_classes is essential.
    # hijja_class_names = [...] # List if needed

    # 5. Feature Scaling
    print("\nScaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 6. Reshape Data for CNN (Assume feature count matches)
    print(f"\nReshaping features ({IMG_DIM}x{IMG_DIM}x{NUM_CHANNELS})...")
    X_train_processed = X_train_scaled.reshape(-1, IMG_DIM, IMG_DIM, NUM_CHANNELS).astype('float32')
    X_test_processed = X_test_scaled.reshape(-1, IMG_DIM, IMG_DIM, NUM_CHANNELS).astype('float32')
    print(f"Data reshaped: Train={X_train_processed.shape}, Test={X_test_processed.shape}")

    # 7. One-Hot Encode Labels
    print("\nOne-hot encoding labels...")
    y_train_cat = tf.keras.utils.to_categorical(y_train_int_labels, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test_int_labels, num_classes=num_classes)
    print(f"Labels one-hot encoded: Train={y_train_cat.shape}, Test={y_test_cat.shape}")

    # 8. Train/Validation Split
    print("\nSplitting into Train/Validation sets (90/10)...")
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_processed, y_train_cat, test_size=0.1, random_state=seed, shuffle=True, stratify=None # Non-stratified simpler
    )
    print(f"Split complete: TrainX={X_train_split.shape}, ValX={X_val_split.shape}")

    data_loaded = True # Flag success
    print("\n--- Simplified Data Handling Complete ---")

except Exception as e:
    print(f"\nERROR during Simplified Data Handling: {e}")
    print("Check assumptions listed in the data loading preamble.")
    # traceback.print_exc() # Uncomment for detailed debug
    data_loaded = False


# ==============================================================================
# 3. HELPER FUNCTIONS: FLOPS ESTIMATION, FORMATTING
# ==============================================================================

def get_flops_simplified(model):
    """Simplified FLOPs estimation."""
    if convert_variables_to_constants_v2 is None: return None
    if not hasattr(model, 'inputs') or not model.inputs: return None
    print("Estimating FLOPs...")
    total_flops = None
    try:
        input_signature = [tf.TensorSpec([1] + inp.shape[1:], inp.dtype) for inp in model.inputs]
        concrete_func = tf.function(model).get_concrete_function(input_signature)
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            # pylint: disable=E1101
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops_profile = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            # pylint: enable=E1101
        total_flops = flops_profile.total_float_ops if hasattr(flops_profile, 'total_float_ops') else None
        print("FLOPs estimation successful." if total_flops else "FLOPs estimation failed/unavailable.")
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
    return total_flops

def format_large_number(number):
    """Formats numbers with K, M, G, T suffixes."""
    if number is None or isinstance(number, str) or np.isnan(number): return "N/A"
    num = float(number); abs_num = abs(num)
    if abs_num < 1e3: return f"{num:,.0f}"
    if abs_num < 1e6: return f"{num / 1e3:.2f} K"
    if abs_num < 1e9: return f"{num / 1e6:.2f} M"
    if abs_num < 1e12: return f"{num / 1e9:.2f} G"
    return f"{num / 1e12:.2f} T"

# ==============================================================================
# 4. ATTENTION MECHANISMS & MODEL BUILDERS
# ==============================================================================

def squeeze_excite_block(input_tensor, ratio=16, name_prefix="se"):
    """Squeeze-and-Excitation block implementation."""
    input_channels = input_tensor.shape[-1]
    if input_channels is None: raise ValueError(f"Channel dim needed for SE block '{name_prefix}'")
    reduced_channels = max(1, int(input_channels / ratio))
    se = GlobalAveragePooling2D(name=f"{name_prefix}_squeeze")(input_tensor)
    se = Reshape((1, 1, input_channels), name=f"{name_prefix}_reshape")(se)
    se = Dense(reduced_channels, activation='relu', use_bias=False, name=f"{name_prefix}_excite1")(se)
    se = Dense(input_channels, activation='sigmoid', use_bias=False, name=f"{name_prefix}_excite2")(se)
    return Multiply(name=f"{name_prefix}_scale")([input_tensor, se])

def build_cnn_with_attention(
    input_shape_local, num_classes_local, dropout_rate,
    attention_module=None, attention_args={},
    attention_placement='after_activation', # 'before_activation' or 'after_activation'
    model_base_name="CNN_Attention"
):
    """Core builder for 3-stage CNN with optional, placement-flexible attention."""
    if attention_module and attention_placement not in ['before_activation', 'after_activation']:
        raise ValueError("Invalid attention_placement value")

    # Construct model name based on config
    attn_tag = ""
    if attention_module:
        attn_name_short = attention_module.__name__.replace('_block','').upper()
        placement_tag = "AfterAct" if attention_placement == 'after_activation' else "BeforeAct"
        # Include ratio in name if present in args
        ratio_tag = f"R{attention_args.get('ratio', 'NA')}" if 'ratio' in attention_args else ""
        attn_tag = f"_{attn_name_short}{ratio_tag}_{placement_tag}"
    model_name = f"{model_base_name}{attn_tag}"

    print(f"Building model: '{model_name}'...")

    inputs = Input(shape=input_shape_local, name="Image_Input")
    x = inputs
    filters_per_stage = [64, 128, 256]

    # Build Convolutional Stages
    for i, filters in enumerate(filters_per_stage):
        stage = i + 1
        x = Conv2D(filters, (3, 3), padding='same', name=f"conv{stage}")(x)
        x = BatchNormalization(name=f"bn{stage}")(x)
        # Apply Attention BEFORE Activation
        if attention_module and attention_placement == 'before_activation':
            x = attention_module(x, name_prefix=f"s{stage}_attn_pre", **attention_args)
        x = Activation('relu', name=f"relu{stage}")(x)
        # Apply Attention AFTER Activation
        if attention_module and attention_placement == 'after_activation':
            x = attention_module(x, name_prefix=f"s{stage}_attn_post", **attention_args)
        x = MaxPooling2D(pool_size=(2, 2), name=f"pool{stage}")(x)
        if dropout_rate > 0: x = Dropout(dropout_rate, name=f"drop{stage}")(x)

    # Classifier
    x = GlobalAveragePooling2D(name="global_pool")(x)
    outputs = Dense(num_classes_local, activation='softmax', dtype='float32', name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Model '{model_name}' built and compiled.")
    return model

# --- Specific Model Instantiation Functions ---
def build_baseline_cnn_local(input_shape_local, num_classes_local, dropout_rate):
    return build_cnn_with_attention(
        input_shape_local, num_classes_local, dropout_rate,
        attention_module=None, model_base_name="Baseline_CNN"
    )

def build_se_cnn_local(input_shape_local, num_classes_local, dropout_rate, se_ratio=16, placement='after_activation'):
    return build_cnn_with_attention(
        input_shape_local, num_classes_local, dropout_rate,
        attention_module=squeeze_excite_block,
        attention_args={'ratio': se_ratio},
        attention_placement=placement,
        model_base_name="SE_CNN" # Name constructed inside core builder
    )

# ==============================================================================
# 5. EXPERIMENT RUNNER FUNCTION (Simplified)
# ==============================================================================

def run_experiment(model, experiment_name, # Pass the expected model name
                   train_data, val_data, test_data):
    """Trains, evaluates, times inference. Returns results dict."""
    print(f"\n--- Running Experiment: {experiment_name} ---")
    (X_train, y_train) = train_data
    (X_val, y_val) = val_data
    (X_test, y_test) = test_data
    start_run_time = time.time()

    results = {'accuracy': 'Error', 'loss': 'Error', 'training_time_s': 'Error',
               'avg_inference_ms': 'Error', 'stopped_epoch': -1}
    history_obj = None

    # Setup Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE_LR, min_lr=1e-7, verbose=1)
    callbacks = [early_stopping, reduce_lr]
    print(f"Callbacks: ES(p={PATIENCE_ES}), ReduceLR(p={PATIENCE_LR})")

    # Train Model
    print(f"Training for max {EPOCHS} epochs (batch_size={BATCH_SIZE})...")
    train_start_time = time.time()
    try:
        history_obj = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=EPOCHS, batch_size=BATCH_SIZE,
                                callbacks=callbacks, shuffle=True, verbose=0) # Verbose 0 for clean logs
        results['training_time_s'] = time.time() - train_start_time
        print(f"Training done ({results['training_time_s']:.1f}s).")

        stopped_epoch = early_stopping.stopped_epoch + 1 if hasattr(early_stopping, 'stopped_epoch') and early_stopping.stopped_epoch > 0 else EPOCHS
        results['stopped_epoch'] = stopped_epoch
        print(f"Stopped at epoch: {stopped_epoch}")

        # Evaluate (uses best weights restored by ES)
        print("Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
        results['accuracy'] = test_accuracy; results['loss'] = test_loss
        print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Measure Inference Time
        print("Measuring inference time...")
        inf_start = time.time()
        _ = model.predict(X_test, batch_size=min(BATCH_SIZE * 4, len(X_test)), verbose=0)
        inf_time = time.time() - inf_start
        results['avg_inference_ms'] = (inf_time / len(X_test)) * 1000 if len(X_test) > 0 else 0
        print(f"  Avg. Inference: {results['avg_inference_ms']:.3f} ms/sample")

    except Exception as e:
        print(f"\nERROR during training/evaluation for '{experiment_name}': {e}")
        if 'train_start_time' in locals(): results['training_time_s'] = time.time() - train_start_time
        # Keep default 'Error' strings for other metrics

    print(f"--- Finished Experiment: {experiment_name} (Total time: {time.time() - start_run_time:.1f}s) ---")
    return results

# ==============================================================================
# 6. MAIN EXECUTION: DEFINE EXPERIMENTS AND RUN
# ==============================================================================

print("\n" + "="*70); print(" Starting SE Attention Placement Comparison "); print("="*70 + "\n")

if data_loaded:
    input_shape = (IMG_DIM, IMG_DIM, NUM_CHANNELS)
    all_experiment_results = {}
    all_complexity_results = {}

    # --- Define Experiment Configurations ---
    experiments_list = [
        {"key": "1_Baseline",      "builder": build_baseline_cnn_local, "args": {}},
        {"key": "2_SE_R32_After",  "builder": build_se_cnn_local, "args": {"se_ratio": 32, "placement": "after_activation"}},
        {"key": "3_SE_R16_After",  "builder": build_se_cnn_local, "args": {"se_ratio": 16, "placement": "after_activation"}},
        {"key": "4_SE_R08_After",  "builder": build_se_cnn_local, "args": {"se_ratio":  8, "placement": "after_activation"}},
        {"key": "5_SE_R04_After",  "builder": build_se_cnn_local, "args": {"se_ratio":  4, "placement": "after_activation"}},
        {"key": "6_SE_R32_Before", "builder": build_se_cnn_local, "args": {"se_ratio": 32, "placement": "before_activation"}},
        {"key": "7_SE_R16_Before", "builder": build_se_cnn_local, "args": {"se_ratio": 16, "placement": "before_activation"}},
        {"key": "8_SE_R08_Before", "builder": build_se_cnn_local, "args": {"se_ratio":  8, "placement": "before_activation"}},
        {"key": "9_SE_R04_Before", "builder": build_se_cnn_local, "args": {"se_ratio":  4, "placement": "before_activation"}},
    ]

    # --- Run Experiments ---
    total_start_time = time.time()
    for i, exp_config in enumerate(experiments_list):
        exp_key = exp_config['key']
        print(f"\n{'='*25} Experiment {i+1}/{len(experiments_list)}: {exp_key} {'='*25}")

        # Reset state
        tf.keras.backend.clear_session(); gc.collect()
        os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
        print(f"Session cleared, seeds reset ({seed}).")

        model_instance = None
        try:
            # Build
            print(f"Building model for {exp_key}...")
            model_instance = exp_config['builder'](
                input_shape_local=input_shape, num_classes_local=num_classes,
                dropout_rate=FIXED_DROPOUT_RATE, **exp_config['args']
            )
            tf_model_name = model_instance.name # Get name assigned by builder

            # Complexity
            params = model_instance.count_params()
            flops = get_flops_simplified(model_instance)
            all_complexity_results[exp_key] = {'params': params, 'flops': flops, 'tf_name': tf_model_name}
            print(f"\nComplexity: Params={format_large_number(params)}, FLOPs={format_large_number(flops)}")

            # Optional Summary: model_instance.summary()

            # Run
            results = run_experiment(
                model=model_instance, experiment_name=tf_model_name,
                train_data=(X_train_split, y_train_split),
                val_data=(X_val_split, y_val_split),
                test_data=(X_test_processed, y_test_cat) # Use processed test data
            )
            all_experiment_results[exp_key] = results

        except Exception as run_err:
            print(f"\nCRITICAL ERROR during experiment '{exp_key}': {run_err}")
            # Store error placeholders
            all_experiment_results[exp_key] = {'accuracy': 'Run Error', 'loss': 'Error', 'training_time_s': 'Error',
                                             'avg_inference_ms': 'Error', 'stopped_epoch': -1}
            if exp_key not in all_complexity_results:
                 all_complexity_results[exp_key] = {'params': 'Error', 'flops': 'Error', 'tf_name': exp_key}
            # traceback.print_exc() # Uncomment for debug

        finally:
            if 'model_instance' in locals() and model_instance is not None: del model_instance
            gc.collect() # Explicit cleanup
            print(f"--- End of Experiment {i+1} ---")

    print(f"\nTotal experiment execution time: {time.time() - total_start_time:.1f} seconds.")

    # --- Final Summary Table ---
    print("\n" + "="*30 + " Experiment Summary: SE Placement Comparison " + "="*30)
    hdr = "{:<20} {:<10} {:<9} {:<9} {:<14} {:<16} {:<8}"
    row = "{:<20} {:<10.4f} {:<9} {:<9} {:<14.1f} {:<16.2f} {:<8}"
    err_row = "{:<20} {:<10} {:<9} {:<9} {:<14} {:<16} {:<8}"
    print(hdr.format("Experiment", "Test Acc", "Params", "FLOPs", "Train Time(s)", "Infer Time(ms)", "Epochs"))
    print("-" * 100)

    sorted_keys = sorted(all_experiment_results.keys(), key=lambda k: int(k.split('_')[0]))
    for key in sorted_keys:
        res = all_experiment_results.get(key, {})
        comp = all_complexity_results.get(key, {})
        # Check if accuracy is valid number before formatting
        if isinstance(res.get('accuracy'), (float, int)):
            print(row.format(
                key,
                res['accuracy'],
                format_large_number(comp.get('params')),
                format_large_number(comp.get('flops')),
                res.get('training_time_s', -1.0), # Provide default if missing
                res.get('avg_inference_ms', -1.0),
                res.get('stopped_epoch', -1)
            ))
        else: # Handle error cases
             print(err_row.format(
                key,
                str(res.get('accuracy', 'Error')),
                format_large_number(comp.get('params', 'Error')),
                format_large_number(comp.get('flops', 'Error')),
                str(res.get('training_time_s', 'Error')),
                str(res.get('avg_inference_ms', 'Error')),
                str(res.get('stopped_epoch', 'Error'))
            ))
    print("-" * 100)
    print("\nNotes: R<N>=SE Reduction Ratio, After/Before=Placement relative to Activation.")
    print(f"       ES Patience={PATIENCE_ES}, LR Patience={PATIENCE_LR}, Dropout={FIXED_DROPOUT_RATE}.")
    print("=" * 100)

else:
    print("\nExecution Halted: Data not loaded successfully.")

