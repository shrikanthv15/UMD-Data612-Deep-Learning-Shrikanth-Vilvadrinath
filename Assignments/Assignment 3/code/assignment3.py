"""
DATA-MSML 612 -- Deep Learning -- Assignment 3
LSTM Sequence Prediction: Learning the Alphabet

Student: Shrikanth Vilvadrinath
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# -- Reproducibility ----------------------------------------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# -- Output directory ---------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def reset_seeds():
    """Reset all random seeds for reproducibility between experiments."""
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)


# =============================================================================
# STEP 1 -- Import Libraries & Define Dataset
# =============================================================================
print("=" * 70)
print("STEP 1 -- Import Libraries & Define Dataset")
print("=" * 70)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
print(f"\nAlphabet: {alphabet}")
print(f"Length:   {len(alphabet)} characters")

# =============================================================================
# STEP 2 -- Character-Integer Mapping
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2 -- Character-Integer Mapping")
print("=" * 70)

char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for i, c in enumerate(alphabet)}

print("\nChar -> Int mapping:")
for c in alphabet:
    print(f"  '{c}' -> {char_to_int[c]}", end="")
    if (char_to_int[c] + 1) % 9 == 0:
        print()
print()

print("Int -> Char mapping:")
for i in range(len(alphabet)):
    print(f"  {i:2d} -> '{int_to_char[i]}'", end="")
    if (i + 1) % 9 == 0:
        print()
print()

# =============================================================================
# STEP 3 -- Prepare Input-Output Pairs
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3 -- Prepare Input-Output Pairs")
print("=" * 70)

dataX = [char_to_int[alphabet[i]] for i in range(25)]
dataY = [char_to_int[alphabet[i + 1]] for i in range(25)]

print(f"\n25 input-output pairs (letter -> next letter):")
for i in range(25):
    print(f"  {alphabet[i]} ({dataX[i]:2d}) -> {alphabet[i+1]} ({dataY[i]:2d})")

# =============================================================================
# STEP 4 -- Reshape X
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4 -- Reshape X to [samples, timesteps, features]")
print("=" * 70)

X = np.array(dataX).reshape(25, 1, 1)
print(f"\nX shape: {X.shape}  (25 samples, 1 timestep, 1 feature)")
print(f"X dtype: {X.dtype}")

# =============================================================================
# STEP 5 -- Normalize
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5 -- Normalize Input Data")
print("=" * 70)

X = X / float(len(alphabet) - 1)  # scale to [0.0, 1.0]
print(f"\nNormalized X by dividing by {len(alphabet) - 1} (number of intervals)")
print(f"X range: [{X.min():.4f}, {X.max():.4f}]")
print(f"\nFirst 5 normalized values:")
for i in range(5):
    print(f"  {alphabet[i]} -> {X[i, 0, 0]:.4f}")

# =============================================================================
# STEP 6 -- One-Hot Encode Output
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6 -- One-Hot Encode Output Variable")
print("=" * 70)

y = to_categorical(dataY, num_classes=26)
print(f"\ny shape: {y.shape}  (25 samples, 26 classes)")
print(f"\nFirst 3 one-hot vectors:")
for i in range(3):
    hot_idx = np.argmax(y[i])
    print(f"  {alphabet[i]}->{alphabet[i+1]}: class {hot_idx} "
          f"({int_to_char[hot_idx]})")

# =============================================================================
# STEP 7 -- Create and Fit Baseline Model
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7 -- Create and Fit Baseline LSTM Model")
print("=" * 70)

reset_seeds()

model = Sequential([
    LSTM(32, input_shape=(1, 1)),
    Dense(26, activation='softmax'),
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy'],
)

print("\nBaseline Model Architecture:")
model.summary()

print(f"\nTraining baseline model (500 epochs, batch_size=1)...")
history = model.fit(X, y, epochs=500, batch_size=1, verbose=0)

print(f"  Training complete.")

# =============================================================================
# STEP 8 -- Print Accuracy
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8 -- Evaluate Baseline Model Accuracy")
print("=" * 70)

loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\n  Loss:     {loss:.4f}")
print(f"  Accuracy: {accuracy * 100:.2f}%")
print(f"\n  Target: >80%")
print(f"  Result: {'PASS' if accuracy >= 0.80 else 'FAIL'} "
      f"({accuracy * 100:.2f}%)")

# =============================================================================
# STEP 9 -- Demonstrate Predictions
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9 -- Demonstrate Predictions")
print("=" * 70)

print(f"\n  {'Input':<8} | {'Expected':<10} | {'Predicted':<10} | {'Correct'}")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

predictions = []
correct_count = 0
for i in range(25):
    x_single = X[i].reshape(1, 1, 1)
    pred = model.predict(x_single, verbose=0)
    pred_idx = np.argmax(pred)
    expected_idx = dataY[i]
    is_correct = pred_idx == expected_idx
    if is_correct:
        correct_count += 1
    predictions.append((alphabet[i], alphabet[expected_idx],
                        int_to_char[pred_idx], is_correct))
    print(f"  {alphabet[i]:<8} | {alphabet[expected_idx]:<10} | "
          f"{int_to_char[pred_idx]:<10} | "
          f"{'Yes' if is_correct else 'No'}")

print(f"\n  Correct: {correct_count}/25 ({correct_count / 25 * 100:.1f}%)")

# Save baseline results + prediction table for audit trail
_bpath = os.path.join(OUTPUT_DIR, 'baseline_results.txt')
with open(_bpath, 'w') as f:
    f.write("Baseline LSTM Results\n")
    f.write("Config: hidden=32, layers=1, lr=0.01, epochs=500, batch=1, seed=42\n")
    f.write("=" * 60 + "\n")
    f.write(f"Loss:     {loss:.4f}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Correct:  {correct_count}/25\n")
    f.write(f"Target:   >80%\n")
    f.write(f"Result:   {'PASS' if accuracy >= 0.80 else 'FAIL'}\n\n")
    f.write("Step 9 -- Full Prediction Table:\n")
    f.write(f"{'Input':<8} | {'Expected':<10} | {'Predicted':<10} | Correct\n")
    f.write(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+---------\n")
    for inp, exp, prd, ok in predictions:
        f.write(f"{inp:<8} | {exp:<10} | {prd:<10} | {'Yes' if ok else 'No'}\n")
    f.write(f"\nTotal: {correct_count}/25 correct ({correct_count/25*100:.1f}%)\n")
print(f"  Saved: output/baseline_results.txt")

# =============================================================================
# STEP 10 -- Depth Experiments: Convergence Analysis
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10 -- Depth Experiments: Convergence Analysis (Stacked LSTMs)")
print("=" * 70)
print("\n  Per the assignment spec, experiments begin with a 2-layer stacked LSTM")
print("  and vary the number of layers. Depth 1 is included as a reference baseline.")
print("  Purpose: convergence analysis -- accuracy/loss curves over 500 epochs.")
print("  (For final-accuracy hyperparameter selection, see Step 11C.)")


def build_lstm_model(num_layers=1, hidden_units=32, learning_rate=0.01):
    """Build an LSTM model with variable depth."""
    model = Sequential()
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        if i == 0:
            model.add(LSTM(hidden_units, input_shape=(1, 1),
                           return_sequences=return_seq))
        else:
            model.add(LSTM(hidden_units, return_sequences=return_seq))
    model.add(Dense(26, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy'],
    )
    return model


DEPTHS = [1, 2, 3, 4]
depth_histories = {}
depth_results = []

for d in DEPTHS:
    print(f"\n  Training {d}-layer LSTM (hidden=32, lr=0.01, 500 epochs)...",
          end=" ", flush=True)
    reset_seeds()
    m = build_lstm_model(num_layers=d, hidden_units=32, learning_rate=0.01)
    h = m.fit(X, y, epochs=500, batch_size=1, verbose=0)
    loss_val, acc_val = m.evaluate(X, y, verbose=0)
    depth_histories[d] = h.history
    depth_results.append((d, acc_val, loss_val))
    print(f"Accuracy = {acc_val * 100:.2f}%, Loss = {loss_val:.4f}")

# Print summary table
print(f"\n  --- Step 10 Summary ---")
print(f"  {'Depth':<8} | {'Accuracy':>10} | {'Loss':>10}")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}")
for d, acc, loss_val in depth_results:
    marker = " <- 2-layer (spec anchor)" if d == 2 else (" <- 1-layer (reference)" if d == 1 else "")
    print(f"  {d:<8} | {acc * 100:>9.2f}% | {loss_val:>10.4f}{marker}")

# Save CSV for audit trail
with open(os.path.join(OUTPUT_DIR, 'step10_depth_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['depth', 'accuracy_pct', 'loss'])
    for d, acc, lv in depth_results:
        writer.writerow([d, f'{acc * 100:.2f}', f'{lv:.4f}'])
print(f"  Saved: output/step10_depth_results.csv")

# -- Plot: Depth Accuracy Curves --
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(DEPTHS)))
for i, d in enumerate(DEPTHS):
    ax.plot(depth_histories[d]['accuracy'],
            label=f'{d} LSTM layer{"s" if d > 1 else ""}',
            color=colors[i], linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Accuracy', fontsize=13)
ax.set_title('Step 10: Training Accuracy by LSTM Depth (500 epochs)',
             fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'step10_depth_accuracy.png'), dpi=150)
plt.close(fig)
print(f"\n  Saved: output/step10_depth_accuracy.png")

# -- Plot: Depth Loss Curves --
fig, ax = plt.subplots(figsize=(10, 6))
for i, d in enumerate(DEPTHS):
    ax.plot(depth_histories[d]['loss'],
            label=f'{d} LSTM layer{"s" if d > 1 else ""}',
            color=colors[i], linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Loss (Categorical Cross-Entropy)', fontsize=13)
ax.set_title('Step 10: Training Loss by LSTM Depth (500 epochs)',
             fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'step10_depth_loss.png'), dpi=150)
plt.close(fig)
print(f"  Saved: output/step10_depth_loss.png")

# =============================================================================
# STEP 11 -- Hyperparameter Experiments
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11 -- Hyperparameter Experiments")
print("=" * 70)

BASELINE_HIDDEN = 32
BASELINE_LAYERS = 1
BASELINE_LR = 0.01
BASELINE_EPOCHS = 500
BASELINE_BATCH = 1

# ── Sweep A: Learning Rate ──────────────────────────────────────────────────
print("\n  --- Sweep A: Learning Rate ---")
LR_VALUES = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
lr_results = []

for lr in LR_VALUES:
    print(f"    lr={lr}...", end=" ", flush=True)
    reset_seeds()
    m = build_lstm_model(num_layers=BASELINE_LAYERS,
                         hidden_units=BASELINE_HIDDEN,
                         learning_rate=lr)
    m.fit(X, y, epochs=BASELINE_EPOCHS, batch_size=BASELINE_BATCH, verbose=0)
    loss_val, acc_val = m.evaluate(X, y, verbose=0)
    lr_results.append((lr, acc_val, loss_val))
    print(f"Accuracy = {acc_val * 100:.2f}%")

print(f"\n  {'Learning Rate':<15} | {'Accuracy':>10} | {'Loss':>10}")
print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")
for lr, acc, loss_val in lr_results:
    print(f"  {lr:<15} | {acc * 100:>9.2f}% | {loss_val:>10.4f}")

best_lr = max(lr_results, key=lambda x: x[1])
print(f"\n  Optimal learning rate: {best_lr[0]} "
      f"(accuracy = {best_lr[1] * 100:.2f}%)")
print(f"  Rationale: Highest accuracy among tested values. "
      f"Too-small LRs (0.0001) underfit in 500 epochs; "
      f"too-large LRs (0.1) may overshoot.")

# -- Plot: Learning Rate Bar Chart --
fig, ax = plt.subplots(figsize=(10, 6))
lr_labels = [str(lr) for lr in LR_VALUES]
lr_accs = [acc * 100 for _, acc, _ in lr_results]
bars = ax.bar(lr_labels, lr_accs, color='steelblue', alpha=0.85)
ax.axhline(y=80, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='80% Target')
ax.set_xlabel('Learning Rate', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Step 11A: Effect of Learning Rate on Accuracy', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 110)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'step11_learning_rate.png'), dpi=150)
plt.close(fig)
print(f"  Saved: output/step11_learning_rate.png")
with open(os.path.join(OUTPUT_DIR, 'step11_learning_rate_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['learning_rate', 'accuracy_pct', 'loss'])
    for lr_v, acc, lv in lr_results:
        writer.writerow([lr_v, f'{acc * 100:.2f}', f'{lv:.4f}'])
print(f"  Saved: output/step11_learning_rate_results.csv")

# ── Sweep B: Hidden Size ────────────────────────────────────────────────────
print("\n  --- Sweep B: Hidden Size ---")
HIDDEN_VALUES = [4, 8, 16, 32, 64, 128]
hidden_results = []

for h in HIDDEN_VALUES:
    print(f"    hidden={h}...", end=" ", flush=True)
    reset_seeds()
    m = build_lstm_model(num_layers=BASELINE_LAYERS,
                         hidden_units=h,
                         learning_rate=BASELINE_LR)
    m.fit(X, y, epochs=BASELINE_EPOCHS, batch_size=BASELINE_BATCH, verbose=0)
    loss_val, acc_val = m.evaluate(X, y, verbose=0)
    hidden_results.append((h, acc_val, loss_val))
    print(f"Accuracy = {acc_val * 100:.2f}%")

print(f"\n  {'Hidden Units':<15} | {'Accuracy':>10} | {'Loss':>10}")
print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")
for h, acc, loss_val in hidden_results:
    print(f"  {h:<15} | {acc * 100:>9.2f}% | {loss_val:>10.4f}")

best_hidden = max(hidden_results, key=lambda x: x[1])
print(f"\n  Optimal hidden size: {best_hidden[0]} "
      f"(accuracy = {best_hidden[1] * 100:.2f}%)")
print(f"  Rationale: Highest accuracy. Smaller sizes (4, 8) lack capacity; "
      f"larger sizes provide diminishing returns on this tiny dataset.")

# -- Plot: Hidden Size Bar Chart --
fig, ax = plt.subplots(figsize=(10, 6))
hidden_labels = [str(h) for h in HIDDEN_VALUES]
hidden_accs = [acc * 100 for _, acc, _ in hidden_results]
bars = ax.bar(hidden_labels, hidden_accs, color='coral', alpha=0.85)
ax.axhline(y=80, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='80% Target')
ax.set_xlabel('Hidden Units', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Step 11B: Effect of Hidden Size on Accuracy', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 110)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'step11_hidden_size.png'), dpi=150)
plt.close(fig)
print(f"  Saved: output/step11_hidden_size.png")
with open(os.path.join(OUTPUT_DIR, 'step11_hidden_size_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['hidden_units', 'accuracy_pct', 'loss'])
    for h_v, acc, lv in hidden_results:
        writer.writerow([h_v, f'{acc * 100:.2f}', f'{lv:.4f}'])
print(f"  Saved: output/step11_hidden_size_results.csv")

# ── Sweep C: Number of Layers ───────────────────────────────────────────────
print("\n  --- Sweep C: Number of Layers ---")
print("  Note: Step 10 examined convergence curves (epoch-by-epoch trajectories).")
print("  This sweep reports final accuracy at epoch 500 for hyperparameter selection.")
print("  The overlapping depth range is intentional -- different analytical purpose.")
LAYER_VALUES = [1, 2, 3, 4, 5]
layer_results = []

for n in LAYER_VALUES:
    print(f"    layers={n}...", end=" ", flush=True)
    reset_seeds()
    m = build_lstm_model(num_layers=n,
                         hidden_units=BASELINE_HIDDEN,
                         learning_rate=BASELINE_LR)
    m.fit(X, y, epochs=BASELINE_EPOCHS, batch_size=BASELINE_BATCH, verbose=0)
    loss_val, acc_val = m.evaluate(X, y, verbose=0)
    layer_results.append((n, acc_val, loss_val))
    print(f"Accuracy = {acc_val * 100:.2f}%")

print(f"\n  {'Layers':<15} | {'Accuracy':>10} | {'Loss':>10}")
print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")
for n, acc, loss_val in layer_results:
    print(f"  {n:<15} | {acc * 100:>9.2f}% | {loss_val:>10.4f}")

best_layers = max(layer_results, key=lambda x: x[1])
print(f"\n  Optimal number of layers: {best_layers[0]} "
      f"(accuracy = {best_layers[1] * 100:.2f}%)")
print(f"  Rationale: Highest accuracy. Deeper networks have more parameters "
      f"but only 25 training samples -- overly deep models may struggle "
      f"to converge.")

# -- Plot: Number of Layers Bar Chart --
fig, ax = plt.subplots(figsize=(10, 6))
layer_labels = [str(n) for n in LAYER_VALUES]
layer_accs = [acc * 100 for _, acc, _ in layer_results]
bars = ax.bar(layer_labels, layer_accs, color='darkgreen', alpha=0.85)
ax.axhline(y=80, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='80% Target')
ax.set_xlabel('Number of LSTM Layers', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Step 11C: Effect of Number of Layers on Accuracy', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 110)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'step11_num_layers.png'), dpi=150)
plt.close(fig)
print(f"  Saved: output/step11_num_layers.png")
with open(os.path.join(OUTPUT_DIR, 'step11_num_layers_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['num_layers', 'accuracy_pct', 'loss'])
    for n_v, acc, lv in layer_results:
        writer.writerow([n_v, f'{acc * 100:.2f}', f'{lv:.4f}'])
print(f"  Saved: output/step11_num_layers_results.csv")

# ── Final Summary: Optimal Configuration ────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 11 -- Final Summary: Optimal Configuration")
print("=" * 70)

print(f"\n  Best from each sweep:")
print(f"    Learning Rate:  {best_lr[0]} ({best_lr[1] * 100:.2f}%)")
print(f"    Hidden Size:    {best_hidden[0]} ({best_hidden[1] * 100:.2f}%)")
print(f"    Number Layers:  {best_layers[0]} ({best_layers[1] * 100:.2f}%)")

print(f"\n  Re-training with optimal configuration: "
      f"lr={best_lr[0]}, hidden={best_hidden[0]}, layers={best_layers[0]}")

reset_seeds()
optimal_model = build_lstm_model(
    num_layers=best_layers[0],
    hidden_units=best_hidden[0],
    learning_rate=best_lr[0],
)
optimal_model.fit(X, y, epochs=BASELINE_EPOCHS, batch_size=BASELINE_BATCH,
                  verbose=0)
opt_loss, opt_acc = optimal_model.evaluate(X, y, verbose=0)

print(f"\n  Optimal Model Results:")
print(f"    Loss:     {opt_loss:.4f}")
print(f"    Accuracy: {opt_acc * 100:.2f}%")
print(f"    Target:   >80%")
print(f"    Result:   {'PASS' if opt_acc >= 0.80 else 'FAIL'}")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 70)
print("DONE -- Assignment 3 Complete")
print("=" * 70)
print(f"\nGenerated plots:")
print(f"  1. output/step10_depth_accuracy.png")
print(f"  2. output/step10_depth_loss.png")
print(f"  3. output/step11_learning_rate.png")
print(f"  4. output/step11_hidden_size.png")
print(f"  5. output/step11_num_layers.png")
print(f"\nGenerated result tables (audit trail):")
print(f"  6. output/baseline_results.txt")
print(f"  7. output/step10_depth_results.csv")
print(f"  8. output/step11_learning_rate_results.csv")
print(f"  9. output/step11_hidden_size_results.csv")
print(f" 10. output/step11_num_layers_results.csv")
print(f"\nBaseline accuracy: {accuracy * 100:.2f}% "
      f"({'PASS' if accuracy >= 0.80 else 'BELOW TARGET'} -- target >80%)")
print(f"Optimal accuracy:  {opt_acc * 100:.2f}% "
      f"({'PASS' if opt_acc >= 0.80 else 'BELOW TARGET'} -- target >80%)")
