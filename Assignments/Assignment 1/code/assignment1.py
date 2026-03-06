"""
DATA-MSML 612 — Deep Learning — Assignment 1
Boston Housing Price Prediction with Multilayer Perceptrons (MLP)

This script implements all 7 steps of the assignment:
  1. Import libraries and load dataset
  2. Split into X (features) and Y (target)
  3. Define baseline model (2+ hidden layers)
  4. Standardize via Pipeline
  5. K-fold cross-validation
  6. Single hidden layer with varying units
  7. Arbitrary depth network experiments
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF info/warning messages

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.initializers import HeNormal
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# STEP 1 — Import Libraries & Load Data
# ==============================================================================
print("=" * 70)
print("STEP 1: Load Dataset")
print("=" * 70)

datafile = os.path.join(DATA_DIR, "housing dataset.csv")
dataset = pd.read_csv(datafile, sep=r'\s+', header=None)
print(f"Dataset shape: {dataset.shape}")
print(f"First 5 rows:\n{dataset.head()}")

# Column names for reference (not in the CSV)
# 0:CRIM, 1:ZN, 2:INDUS, 3:CHAS, 4:NOX, 5:RM, 6:AGE, 7:DIS,
# 8:RAD, 9:TAX, 10:PTRATIO, 11:B, 12:LSTAT, 13:MEDV (target)

# ==============================================================================
# STEP 2 — Split X and Y
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Split into X (features) and Y (target)")
print("=" * 70)

data = dataset.values
X = data[:, 0:13]
Y = data[:, 13]

print(f"X shape: {X.shape}  (13 features)")
print(f"Y shape: {Y.shape}  (target: MEDV — median home value in $1000s)")
print(f"Y range: [{Y.min():.1f}, {Y.max():.1f}], mean={Y.mean():.2f}")

# ==============================================================================
# STEP 3 — Define Baseline Model (2+ hidden layers)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Define Baseline Model")
print("=" * 70)


def build_baseline_model():
    """
    Baseline MLP: Input(13) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1)
    - ReLU activation: avoids vanishing gradient problem
    - He normal init: recommended for ReLU activations
    - Adam optimizer: adaptive learning rate, robust default
    - MSE loss: standard for regression
    """
    model = Sequential([
        InputLayer(input_shape=(13,)),
        Dense(64, activation='relu', kernel_initializer=HeNormal(seed=SEED)),
        Dense(32, activation='relu', kernel_initializer=HeNormal(seed=SEED)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Print model summary
print("Baseline model architecture:")
temp_model = build_baseline_model()
temp_model.summary()
del temp_model

# ==============================================================================
# STEP 4 — Standardize via Pipeline
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Standardize via Pipeline")
print("=" * 70)
print("Using sklearn Pipeline with StandardScaler + KerasRegressor.")
print("This ensures standardization is applied INSIDE each CV fold,")
print("preventing data leakage from the test fold into the training fold.")

# ==============================================================================
# STEP 5 — K-Fold Cross-Validation
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Evaluate Baseline Model with 10-Fold Cross-Validation")
print("=" * 70)

EPOCHS = 100
BATCH_SIZE = 16
K = 10

estimator = KerasRegressor(model=build_baseline_model, epochs=EPOCHS,
                           batch_size=BATCH_SIZE, verbose=0)
pipeline = Pipeline([
    ('standardize', StandardScaler()),
    ('mlp', estimator)
])

kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)
results = cross_val_score(pipeline, X, Y, cv=kfold,
                          scoring='neg_mean_squared_error')

# sklearn returns negative MSE (to maximize); negate for actual MSE
mse_scores = -results
print(f"\nBaseline Results ({K}-fold CV, {EPOCHS} epochs, batch_size={BATCH_SIZE}):")
print(f"  MSE per fold: {np.round(mse_scores, 2)}")
print(f"  Mean MSE:     {mse_scores.mean():.2f}")
print(f"  Std MSE:      {mse_scores.std():.2f}")

baseline_mean = mse_scores.mean()
baseline_std = mse_scores.std()

# ==============================================================================
# STEP 6 — Single Hidden Layer with Varying Units
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Single Hidden Layer — Effect of Varying Hidden Units")
print("=" * 70)

HIDDEN_UNITS = [1, 5, 10, 20, 50, 100, 200]
step6_results = []


def build_single_layer_model(hidden_units=13):
    """Single hidden layer: Input(13) -> Dense(N, ReLU) -> Dense(1)"""
    model = Sequential([
        InputLayer(input_shape=(13,)),
        Dense(hidden_units, activation='relu',
              kernel_initializer=HeNormal(seed=SEED)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


for n_units in HIDDEN_UNITS:
    print(f"\n  Testing hidden_units={n_units}...", end=" ", flush=True)

    estimator = KerasRegressor(
        model=build_single_layer_model,
        model__hidden_units=n_units,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    pipeline = Pipeline([
        ('standardize', StandardScaler()),
        ('mlp', estimator)
    ])

    kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X, Y, cv=kfold,
                             scoring='neg_mean_squared_error')
    mse = -scores
    mean_mse = mse.mean()
    std_mse = mse.std()
    step6_results.append((n_units, mean_mse, std_mse))
    print(f"Mean MSE = {mean_mse:.2f} (+/- {std_mse:.2f})")

# Print summary table
print("\n  --- Step 6 Summary ---")
print(f"  {'Hidden Units':>12} | {'Mean MSE':>10} | {'Std MSE':>10}")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}")
for units, mean, std in step6_results:
    print(f"  {units:>12} | {mean:>10.2f} | {std:>10.2f}")

# Plot: Hidden Units vs MSE
units_list = [r[0] for r in step6_results]
means_list = [r[1] for r in step6_results]
stds_list = [r[2] for r in step6_results]

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(units_list, means_list, yerr=stds_list, fmt='o-', capsize=5,
            capthick=2, linewidth=2, markersize=8, color='steelblue',
            ecolor='coral')
ax.set_xlabel('Number of Hidden Units', fontsize=13)
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13)
ax.set_title('Step 6: Effect of Hidden Layer Size on MSE\n'
             f'(Single Hidden Layer, {K}-fold CV, {EPOCHS} epochs)',
             fontsize=14)
ax.set_xscale('log')
ax.set_xticks(units_list)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'step6_hidden_units.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n  Plot saved: {plot_path}")

# ==============================================================================
# STEP 7 — Arbitrary Depth Network
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Arbitrary Depth Network — Effect of Depth on MSE")
print("=" * 70)

DEPTHS = [1, 2, 3, 4, 5, 8]
WIDTH = 32
EPOCHS_DEEP = 150
step7_results = []


def build_deep_model(depth=1):
    """
    Variable-depth MLP: Input(13) -> [Dense(32, ReLU)] x depth -> Dense(1)
    Width fixed at 32 to isolate the effect of depth.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(13,)))
    for _ in range(depth):
        model.add(Dense(WIDTH, activation='relu',
                        kernel_initializer=HeNormal(seed=SEED)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


for d in DEPTHS:
    print(f"\n  Testing depth={d}...", end=" ", flush=True)

    estimator = KerasRegressor(
        model=build_deep_model,
        model__depth=d,
        epochs=EPOCHS_DEEP,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    pipeline = Pipeline([
        ('standardize', StandardScaler()),
        ('mlp', estimator)
    ])

    kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X, Y, cv=kfold,
                             scoring='neg_mean_squared_error')
    mse = -scores
    mean_mse = mse.mean()
    std_mse = mse.std()
    step7_results.append((d, mean_mse, std_mse))
    print(f"Mean MSE = {mean_mse:.2f} (+/- {std_mse:.2f})")

# Print summary table
print("\n  --- Step 7 Summary ---")
print(f"  {'Depth':>6} | {'Mean MSE':>10} | {'Std MSE':>10}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}")
for depth, mean, std in step7_results:
    print(f"  {depth:>6} | {mean:>10.2f} | {std:>10.2f}")

# Plot: Depth vs MSE
depths_list = [r[0] for r in step7_results]
means_list = [r[1] for r in step7_results]
stds_list = [r[2] for r in step7_results]

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(depths_list, means_list, yerr=stds_list, fmt='s-', capsize=5,
            capthick=2, linewidth=2, markersize=8, color='darkgreen',
            ecolor='orange')
ax.set_xlabel('Number of Hidden Layers (Depth)', fontsize=13)
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13)
ax.set_title('Step 7: Effect of Network Depth on MSE\n'
             f'(Width={WIDTH}, {K}-fold CV, {EPOCHS_DEEP} epochs)',
             fontsize=14)
ax.set_xticks(depths_list)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'step7_depth.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n  Plot saved: {plot_path}")

# --- Optional: Convergence curves (training loss vs epoch by depth) ---
print("\n  Generating convergence curves...")

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(DEPTHS)))

for i, d in enumerate(DEPTHS):
    print(f"    Training depth={d} with validation split...", end=" ", flush=True)

    # Standardize for convergence plot (single train/val split)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = build_deep_model(depth=d)
    history = model.fit(X_scaled, Y, epochs=EPOCHS_DEEP, batch_size=BATCH_SIZE,
                        validation_split=0.2, verbose=0)

    ax.plot(history.history['loss'], label=f'Depth {d} (train)',
            color=colors[i], linewidth=1.5)
    ax.plot(history.history['val_loss'], label=f'Depth {d} (val)',
            color=colors[i], linewidth=1.5, linestyle='--')
    print("done")

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('MSE Loss', fontsize=13)
ax.set_title('Step 7: Training Convergence by Network Depth\n'
             f'(Width={WIDTH}, {EPOCHS_DEEP} epochs, 80/20 train/val split)',
             fontsize=14)
ax.legend(fontsize=9, ncol=2)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'step7_convergence.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  Plot saved: {plot_path}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF ALL EXPERIMENTS")
print("=" * 70)

print(f"\n  Baseline Model (2 hidden layers: 64→32, {EPOCHS} epochs):")
print(f"    Mean MSE = {baseline_mean:.2f} (+/- {baseline_std:.2f})")

print(f"\n  Step 6 — Single Hidden Layer ({EPOCHS} epochs):")
print(f"  {'Hidden Units':>12} | {'Mean MSE':>10} | {'Std MSE':>10}")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}")
for units, mean, std in step6_results:
    print(f"  {units:>12} | {mean:>10.2f} | {std:>10.2f}")

print(f"\n  Step 7 — Variable Depth (width={WIDTH}, {EPOCHS_DEEP} epochs):")
print(f"  {'Depth':>6} | {'Mean MSE':>10} | {'Std MSE':>10}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}")
for depth, mean, std in step7_results:
    print(f"  {depth:>6} | {mean:>10.2f} | {std:>10.2f}")

print("\n" + "=" * 70)
print("DONE — All plots saved to output/ directory.")
print("=" * 70)
