"""Generate demo notebooks 03–08 for pyUDE."""

import json
import os

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

NB_DIR = os.path.dirname(__file__)


def nb(title, cells):
    """Create a notebook with kernelspec metadata."""
    book = new_notebook(cells=cells)
    book.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    book.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return book


def write(name, book):
    path = os.path.join(NB_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(book, f)
    print(f"  wrote {name}")


md = new_markdown_cell
code = new_code_cell

# ===========================================================================
# Notebook 03 — Validation & Model Evaluation
# ===========================================================================

nb03 = nb("03_validation_evaluation", [
    md("# Notebook 03 — Validation & Model Evaluation\n\n"
       "Training loss tells you how well a model memorises its training data — not whether it\n"
       "can predict new observations. This notebook shows how to:\n\n"
       "- Hold out a future test window with `train_test_split`\n"
       "- Track train vs validation loss during training with `val_data`\n"
       "- Use early stopping to prevent overfitting\n"
       "- Evaluate forecasts with `mse`, `rmse`, `mae`, and `r2_score`\n"
       "- Compare `NODE` against `CustomDerivatives` on the same dataset"),

    code("import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import torch\n"
         "from scipy.integrate import solve_ivp\n"
         "import pyUDE as ude\n\n"
         "np.random.seed(42)\n"
         "torch.manual_seed(42)\n"
         "print(f'pyUDE {ude.__version__}')"),

    md("## 1 — Generate Data: Damped Oscillator\n\n"
       "The damped harmonic oscillator has the exact solution we can compare against:\n\n"
       "```\n"
       "dx/dt = y\n"
       "dy/dt = -ω²x - 2γy\n"
       "```\n\n"
       "with `ω = 2.0`, `γ = 0.3`. We will pretend we only know `ω` and must learn `γ` from data."),

    code("omega, gamma = 2.0, 0.3\n\n"
         "def damped_osc(t, u):\n"
         "    x, y = u\n"
         "    return [y, -omega**2 * x - 2*gamma * y]\n\n"
         "sol = solve_ivp(damped_osc, [0, 6], [1.0, 0.0], t_eval=np.linspace(0, 6, 120),\n"
         "                method='RK45', dense_output=False)\n\n"
         "noise_scale = 0.05\n"
         "x_noisy = sol.y[0] + noise_scale * np.random.randn(120)\n"
         "y_noisy = sol.y[1] + noise_scale * np.random.randn(120)\n\n"
         "data = pd.DataFrame({'time': sol.t, 'x': x_noisy, 'y': y_noisy})\n"
         "print(data.shape, data.head(3))"),

    md("## 2 — Train/Test Split\n\n"
       "`train_test_split` always preserves temporal order — it never shuffles rows.\n"
       "The test set covers the *future* part of the trajectory."),

    code("train_data, test_data = ude.train_test_split(data, test_fraction=0.25)\n"
         "print(f'Train: {len(train_data)} rows  |  Test: {len(test_data)} rows')\n\n"
         "fig, ax = plt.subplots(figsize=(10, 3))\n"
         "ax.plot(train_data['time'], train_data['x'], 'b.', ms=3, label='train x')\n"
         "ax.plot(test_data['time'],  test_data['x'],  'r.', ms=3, label='test x')\n"
         "ax.axvline(train_data['time'].iloc[-1], color='k', ls='--', label='split')\n"
         "ax.set_xlabel('time'); ax.legend(); ax.set_title('Temporal train/test split')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 3 — Baseline Model (No Validation)\n\n"
       "Train on `train_data` only. The model has no information about how it performs on\n"
       "the test window — it may overfit or simply not extrapolate well."),

    code("model_base = ude.NODE(train_data, hidden_units=32, hidden_layers=2)\n"
         "model_base.train(loss='derivative_matching', epochs=300, verbose=False)\n\n"
         "fc_base = model_base.forecast(steps=len(test_data))\n\n"
         "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
         "for ax, col in zip(axes, ['x', 'y']):\n"
         "    ax.plot(data['time'], data[col], 'k.', ms=3, alpha=0.4, label='observed')\n"
         "    ax.plot(fc_base['time'], fc_base[col], 'b-', lw=2, label='forecast (baseline)')\n"
         "    ax.axvline(train_data['time'].iloc[-1], color='k', ls='--')\n"
         "    ax.set_title(col); ax.legend(fontsize=8)\n"
         "plt.suptitle('Baseline NODE (no val_data)'); plt.tight_layout(); plt.show()"),

    md("## 4 — Training with `val_data`\n\n"
       "Passing `val_data` causes the trainer to compute validation loss every `val_interval`\n"
       "epochs and store it in `model.train_history_`. This lets us see whether the model is\n"
       "overfitting before training finishes."),

    code("model_val = ude.NODE(train_data, hidden_units=32, hidden_layers=2)\n"
         "model_val.train(\n"
         "    loss='derivative_matching', epochs=500,\n"
         "    val_data=test_data, val_interval=5,\n"
         "    verbose=False,\n"
         ")\n"
         "print('train_history_ keys:', list(model_val.train_history_.keys()))\n"
         "print('val_loss entries:    ', len(model_val.train_history_['val_loss']))"),

    md("### Plot training and validation loss"),

    code("history = model_val.train_history_\n"
         "epochs_train = range(1, len(history['train_loss']) + 1)\n\n"
         "fig, ax = plt.subplots(figsize=(9, 4))\n"
         "ax.semilogy(epochs_train, history['train_loss'], alpha=0.6, label='train loss')\n"
         "ax.semilogy(history['val_epochs'], history['val_loss'], 'r-', lw=2, label='val loss')\n"
         "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (log scale)')\n"
         "ax.legend(); ax.set_title('Train vs Validation Loss')\n"
         "plt.tight_layout(); plt.show()\n\n"
         "print('Final train loss:', f\"{history['train_loss'][-1]:.6f}\")\n"
         "print('Final val loss:  ', f\"{history['val_loss'][-1]:.6f}\")"),

    md("## 5 — Val-Based Early Stopping\n\n"
       "Set `patience=N` to stop training when validation loss does not improve for `N`\n"
       "consecutive val epochs. The **best weights** are automatically restored on stopping."),

    code("model_es = ude.NODE(train_data, hidden_units=32, hidden_layers=2)\n"
         "model_es.train(\n"
         "    loss='derivative_matching', epochs=1000,\n"
         "    val_data=test_data, val_interval=5, patience=30,\n"
         "    verbose=True,\n"
         ")\n"
         "print('Epochs actually run:', len(model_es.train_history_['train_loss']))"),

    md("## 6 — Compute Test-Set Metrics\n\n"
       "All four metric functions return a `dict` with one entry per state column plus a\n"
       "`\"mean\"` key averaging across states."),

    code("fc_es = model_es.forecast(steps=len(test_data))\n"
         "min_len = min(len(fc_es), len(test_data))\n"
         "obs  = test_data.iloc[:min_len].reset_index(drop=True)\n"
         "pred = fc_es.iloc[:min_len].reset_index(drop=True)\n"
         "state_cols = ['x', 'y']\n\n"
         "print('MSE:  ', ude.mse(obs[state_cols],    pred[state_cols]))\n"
         "print('RMSE: ', ude.rmse(obs[state_cols],   pred[state_cols]))\n"
         "print('MAE:  ', ude.mae(obs[state_cols],    pred[state_cols]))\n"
         "print('R²:   ', ude.r2_score(obs[state_cols], pred[state_cols]))"),

    md("### `score()` dispatcher\n\n"
       "Prefer `score()` when the metric is a variable — e.g. when running comparisons in a loop."),

    code("for metric in ['mse', 'rmse', 'mae', 'r2_score']:\n"
         "    s = ude.score(obs[state_cols], pred[state_cols], metric=metric)\n"
         "    print(f'{metric:>10s}  mean={s[\"mean\"]:.4f}   x={s[\"x\"]:.4f}  y={s[\"y\"]:.4f}')"),

    md("## 7 — NODE vs CustomDerivatives\n\n"
       "We supply the known restoring force (`-ω²x`) but let the NN learn the damping term\n"
       "`-2γy`. This injects partial knowledge without requiring the full equation."),

    code("import torch\n\n"
         "def osc_known(u, p, t):\n"
         "    x, y_ = u[0], u[1]\n"
         "    return torch.stack([\n"
         "        y_,                  # dx/dt = y  (known)\n"
         "        -omega**2 * x,       # -ω²x        (known restoring force)\n"
         "        # damping -2γy is unknown — learned by the NN\n"
         "    ])\n\n"
         "model_cd = ude.CustomDerivatives(\n"
         "    train_data, osc_known,\n"
         "    init_params={},   # no mechanistic parameters to learn here\n"
         "    hidden_units=32,\n"
         ")\n"
         "model_cd.train(loss='derivative_matching', epochs=300, verbose=False)\n\n"
         "fc_cd = model_cd.forecast(steps=len(test_data))\n"
         "pred_cd = fc_cd.iloc[:min_len].reset_index(drop=True)\n\n"
         "rmse_node = ude.rmse(obs[state_cols], pred[state_cols])['mean']\n"
         "rmse_cd   = ude.rmse(obs[state_cols], pred_cd[state_cols])['mean']\n\n"
         "print(f'NODE RMSE (mean):              {rmse_node:.4f}')\n"
         "print(f'CustomDerivatives RMSE (mean): {rmse_cd:.4f}')"),

    md("## Key Takeaways\n\n"
       "- **`train_test_split`** preserves temporal order; always split before training.\n"
       "- **`val_data`** in `train()` tracks generalisation loss without modifying the model.\n"
       "- **`train_history_`** persists across multiple `train()` calls and accumulates all phases.\n"
       "- **`patience`** stops training at the best validation checkpoint — not the last epoch.\n"
       "- **`mse`, `rmse`, `mae`, `r2_score`** all accept DataFrames or arrays and return "
       "`{state: val, 'mean': val}`.\n"
       "- Adding partial mechanistic knowledge via `CustomDerivatives` typically improves "
       "generalisation on the test window."),
])

# ===========================================================================
# Notebook 04 — Regularization
# ===========================================================================

nb04 = nb("04_regularization", [
    md("# Notebook 04 — Regularization\n\n"
       "Overfitting is the main failure mode for UDEs trained on small or noisy datasets:\n"
       "the network memorises the training observations and produces a forecast that diverges\n"
       "almost immediately. This notebook demonstrates four regularization tools:\n\n"
       "| Tool | Argument | Effect |\n"
       "|------|----------|--------|\n"
       "| L2 weight decay | `weight_decay=` | Shrinks all weights evenly |\n"
       "| L1 penalty | `lambda_l1=` | Promotes sparse weights |\n"
       "| Elastic Net | both | Combination of L1 + L2 |\n"
       "| Dropout | `dropout=` | Randomly drops neurons during training |\n"
       "| Parameter bounds | `param_bounds=` | Keeps mechanistic params physical |"),

    code("import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import torch\n"
         "from scipy.integrate import solve_ivp\n"
         "import pyUDE as ude\n\n"
         "np.random.seed(0)\n"
         "torch.manual_seed(0)"),

    md("## 1 — Generate Small, Noisy Lotka-Volterra Data\n\n"
       "Only 40 observations with 10% Gaussian noise — a deliberately difficult regime."),

    code("def lv_rhs(t, u, alpha=1.0, beta=0.1, delta=0.075, gamma=1.5):\n"
         "    x, y = u\n"
         "    return [alpha*x - beta*x*y, delta*x*y - gamma*y]\n\n"
         "sol = solve_ivp(lv_rhs, [0, 15], [10.0, 5.0], t_eval=np.linspace(0, 15, 40),\n"
         "                method='RK45')\n\n"
         "noise = 0.10\n"
         "data = pd.DataFrame({\n"
         "    'time':     sol.t,\n"
         "    'prey':     sol.y[0] * (1 + noise * np.random.randn(40)),\n"
         "    'predator': sol.y[1] * (1 + noise * np.random.randn(40)),\n"
         "})\n\n"
         "# Dense true trajectory for comparison\n"
         "sol_true = solve_ivp(lv_rhs, [0, 20], [10.0, 5.0],\n"
         "                     t_eval=np.linspace(0, 20, 400), method='RK45')\n\n"
         "train_data, test_data = ude.train_test_split(data, test_fraction=0.2)\n"
         "print(f'Train: {len(train_data)}  Test: {len(test_data)}')"),

    md("## 2 — Overfitting Baseline\n\n"
       "A large network (`hidden_units=128`) trained without regularization will memorise the\n"
       "training points and produce a forecast that diverges past the training window."),

    code("EPOCHS = 150\n\n"
         "model_base = ude.NODE(train_data, hidden_units=64, hidden_layers=2)\n"
         "model_base.train(loss='derivative_matching', epochs=EPOCHS, verbose=False)\n"
         "fc_base = model_base.forecast(steps=60)\n"
         "print('Baseline trained.')"),

    md("## 3 — L2 Weight Decay\n\n"
       "`weight_decay` is passed directly to the Adam optimizer and applies L2 regularization\n"
       "by adding `λ * ||θ||²` to the loss at each step."),

    code("model_l2 = ude.NODE(train_data, hidden_units=64, hidden_layers=2)\n"
         "model_l2.train(loss='derivative_matching', epochs=EPOCHS,\n"
         "               weight_decay=1e-3, verbose=False)\n"
         "fc_l2 = model_l2.forecast(steps=60)\n"
         "print('L2 trained.')"),

    md("## 4 — L1 Penalty\n\n"
       "`lambda_l1` adds an explicit L1 penalty `λ * Σ|θ|` to the loss. Unlike L2, L1 pushes\n"
       "individual weights to exactly zero, creating sparse networks."),

    code("model_l1 = ude.NODE(train_data, hidden_units=64, hidden_layers=2)\n"
         "model_l1.train(loss='derivative_matching', epochs=EPOCHS,\n"
         "               lambda_l1=1e-4, verbose=False)\n"
         "fc_l1 = model_l1.forecast(steps=60)\n"
         "print('L1 trained.')"),

    md("## 5 — Dropout\n\n"
       "`dropout=0.2` randomly zeros 20% of neurons at each training step, preventing\n"
       "the network from co-adapting to specific patterns in the training data."),

    code("model_do = ude.NODE(train_data, hidden_units=64, hidden_layers=2, dropout=0.2)\n"
         "model_do.train(loss='derivative_matching', epochs=EPOCHS, verbose=False)\n"
         "fc_do = model_do.forecast(steps=60)\n"
         "print('Dropout trained.')"),

    md("## 6 — Side-by-Side Forecast Comparison"),

    code("state_cols = ['prey', 'predator']\n"
         "true_t = sol_true.t\n\n"
         "fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey='row')\n"
         "labels = ['Baseline', 'L2 (1e-3)', 'L1 (1e-4)', 'Dropout (0.2)']\n"
         "forecasts = [fc_base, fc_l2, fc_l1, fc_do]\n\n"
         "for col_idx, (label, fc) in enumerate(zip(labels, forecasts)):\n"
         "    for row_idx, col in enumerate(state_cols):\n"
         "        ax = axes[row_idx, col_idx]\n"
         "        ax.plot(true_t, sol_true.y[row_idx], 'k-', alpha=0.3, lw=1, label='true')\n"
         "        ax.plot(train_data['time'], train_data[col], 'b.', ms=4, label='train')\n"
         "        ax.plot(fc['time'], fc[col], 'r-', lw=2, label='forecast')\n"
         "        ax.axvline(train_data['time'].iloc[-1], color='k', ls='--', lw=1)\n"
         "        if row_idx == 0:\n"
         "            ax.set_title(label, fontsize=11)\n"
         "        if col_idx == 0:\n"
         "            ax.set_ylabel(col)\n\n"
         "axes[0, 0].legend(fontsize=7, loc='upper right')\n"
         "plt.suptitle('Regularization Comparison', fontsize=13)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 7 — RMSE Comparison on Test Set"),

    code("from copy import deepcopy\n\n"
         "# Compute RMSE for each model on the test window\n"
         "min_len = len(test_data)\n"
         "for label, fc in zip(labels, forecasts):\n"
         "    aligned = fc.iloc[:min_len].reset_index(drop=True)\n"
         "    r = ude.rmse(test_data[state_cols].reset_index(drop=True),\n"
         "                 aligned[state_cols])\n"
         "    print(f'{label:<22}  RMSE mean={r[\"mean\"]:.3f}  prey={r[\"prey\"]:.3f}  pred={r[\"predator\"]:.3f}')"),

    md("## 8 — `param_bounds` on CustomDerivatives\n\n"
       "Mechanistic parameters like growth rates must be positive. Without bounds, gradient\n"
       "updates can push them negative, making the model physically nonsensical."),

    code("def lv_known(u, p, t):\n"
         "    \"\"\"Known linear growth / death terms. Predation is learned by the NN.\"\"\"\n"
         "    prey, pred = u[0], u[1]\n"
         "    return torch.stack([\n"
         "        p['alpha'] * prey,    # prey growth (known structure)\n"
         "        -p['gamma'] * pred,   # predator death (known structure)\n"
         "    ])\n\n"
         "# Without bounds — alpha and gamma can go negative\n"
         "model_no_bounds = ude.CustomDerivatives(\n"
         "    train_data, lv_known,\n"
         "    init_params={'alpha': 0.5, 'gamma': 1.0},\n"
         ")\n"
         "model_no_bounds.train(loss='derivative_matching', epochs=300, verbose=False)\n"
         "print('No bounds — learned params:', model_no_bounds.get_params())\n\n"
         "# With bounds — both params stay >= 0\n"
         "model_bounded = ude.CustomDerivatives(\n"
         "    train_data, lv_known,\n"
         "    init_params={'alpha': 0.5, 'gamma': 1.0},\n"
         "    param_bounds={'alpha': (0.0, None), 'gamma': (0.0, None)},\n"
         ")\n"
         "model_bounded.train(loss='derivative_matching', epochs=300, verbose=False)\n"
         "print('Bounded  — learned params:', model_bounded.get_params())"),

    md("## Key Takeaways\n\n"
       "- **L2** (`weight_decay`) shrinks weights uniformly — good all-round regularizer.\n"
       "- **L1** (`lambda_l1`) promotes sparsity — useful when only a few terms drive dynamics.\n"
       "- **Elastic Net** = L1 + L2 simultaneously.\n"
       "- **Dropout** (`dropout=`) works well with small datasets; default `0.2` is a safe start.\n"
       "- **`param_bounds`** is essential for `CustomDerivatives` when parameters have physical\n"
       "  meaning — it prevents the optimizer from pushing them into physically impossible ranges.\n"
       "- All regularizers are backward-compatible: default values (`0.0` / `None`) disable them."),
])

# ===========================================================================
# Notebook 05 — Cross-Validation & Model Selection
# ===========================================================================

nb05 = nb("05_cross_validation", [
    md("# Notebook 05 — Cross-Validation & Model Selection\n\n"
       "A single train/test split is fragile: performance depends on where you drew the\n"
       "boundary. **Expanding-window cross-validation** evaluates the model across multiple\n"
       "consecutive windows, giving a more reliable estimate of generalisation performance\n"
       "and a principled way to select hyperparameters."),

    code("import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import torch\n"
         "from scipy.integrate import solve_ivp\n"
         "import pyUDE as ude\n\n"
         "np.random.seed(7)\n"
         "torch.manual_seed(7)"),

    md("## 1 — Generate Data: Logistic Growth\n\n"
       "A single-state system `dx/dt = r*x*(1 - x/K)` with known analytic solution.\n"
       "Simple enough that a grid search over `hidden_units` produces meaningfully different\n"
       "results."),

    code("def logistic(t, u, r=0.5, K=10.0):\n"
         "    return [r * u[0] * (1 - u[0] / K)]\n\n"
         "sol = solve_ivp(logistic, [0, 20], [0.5], t_eval=np.linspace(0, 20, 150),\n"
         "                method='RK45')\n"
         "data = pd.DataFrame({'time': sol.t,\n"
         "                     'x': sol.y[0] + 0.1 * np.random.randn(150)})\n"
         "print(data.shape)"),

    md("## 2 — Visualise the Expanding Window\n\n"
       "Each fold uses all rows up to a given time for training and the next block for\n"
       "validation. The training window expands left-to-right across the time series."),

    code("n = len(data)\n"
         "min_train = int(0.5 * n)   # 50% minimum training size\n"
         "n_splits  = 4\n"
         "step      = (n - min_train) // n_splits\n\n"
         "fig, ax = plt.subplots(figsize=(12, 2.5))\n"
         "colors = plt.cm.Blues(np.linspace(0.35, 0.85, n_splits))\n"
         "for fold in range(n_splits):\n"
         "    train_end = min_train + fold * step\n"
         "    val_end   = train_end + step if fold < n_splits - 1 else n\n"
         "    ax.barh(fold, train_end,          left=0,         height=0.6,\n"
         "            color=colors[fold], label=f'fold {fold+1} train')\n"
         "    ax.barh(fold, val_end - train_end, left=train_end, height=0.6,\n"
         "            color='salmon', alpha=0.7)\n\n"
         "ax.set_xlabel('Row index'); ax.set_yticks(range(n_splits))\n"
         "ax.set_yticklabels([f'fold {i+1}' for i in range(n_splits)])\n"
         "ax.set_title('Expanding-window folds  (blue = train, red = val)')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 3 — Basic CV Run"),

    code("results = ude.time_series_cv(\n"
         "    data,\n"
         "    model_class=ude.NODE,\n"
         "    model_kwargs={'hidden_units': 32, 'hidden_layers': 2},\n"
         "    train_kwargs={'loss': 'derivative_matching', 'epochs': 80, 'verbose': False},\n"
         "    n_splits=4,\n"
         "    metric='rmse',\n"
         ")\n\n"
         "for r in results:\n"
         "    print(f\"fold {r['fold']}  n_train={r['n_train']:3d}  \"\n"
         "          f\"n_val={r['n_val']}  val_score={r['val_score']:.4f}\")"),

    md("## 4 — Mean ± Std Across Folds"),

    code("scores = [r['val_score'] for r in results]\n"
         "print(f'RMSE: mean={np.mean(scores):.4f}  std={np.std(scores):.4f}')\n\n"
         "fig, ax = plt.subplots(figsize=(6, 3))\n"
         "ax.bar(range(1, len(scores)+1), scores, color='steelblue', alpha=0.8)\n"
         "ax.axhline(np.mean(scores), color='r', ls='--', label=f'mean={np.mean(scores):.4f}')\n"
         "ax.set_xlabel('Fold'); ax.set_ylabel('RMSE'); ax.legend()\n"
         "ax.set_title('Val RMSE per fold')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 5 — Grid Search over `hidden_units`\n\n"
       "Run CV for each configuration and record the mean val RMSE. This is the standard\n"
       "way to choose network width without cheating on the test set."),

    code("configs = [8, 16, 32]\n"
         "cv_scores = {}\n\n"
         "for h in configs:\n"
         "    res = ude.time_series_cv(\n"
         "        data,\n"
         "        model_class=ude.NODE,\n"
         "        model_kwargs={'hidden_units': h, 'hidden_layers': 2},\n"
         "        train_kwargs={'loss': 'derivative_matching', 'epochs': 50, 'verbose': False},\n"
         "        n_splits=4,\n"
         "        metric='rmse',\n"
         "    )\n"
         "    cv_scores[h] = np.mean([r['val_score'] for r in res])\n"
         "    print(f'hidden_units={h:3d}  mean RMSE={cv_scores[h]:.4f}')\n\n"
         "best_h = min(cv_scores, key=cv_scores.get)\n"
         "print(f'\\nBest hidden_units: {best_h}  (RMSE={cv_scores[best_h]:.4f})')"),

    code("fig, ax = plt.subplots(figsize=(7, 4))\n"
         "ax.bar([str(h) for h in configs], [cv_scores[h] for h in configs],\n"
         "       color=['green' if h == best_h else 'steelblue' for h in configs])\n"
         "ax.set_xlabel('hidden_units'); ax.set_ylabel('Mean CV RMSE')\n"
         "ax.set_title('Grid search: hidden_units vs CV RMSE  (green = best)')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 6 — Comparing Loss Functions\n\n"
       "Is `derivative_matching` or `simulation` more accurate on the validation folds?"),

    code("loss_scores = {}\n"
         "for loss_fn in ['derivative_matching', 'simulation']:\n"
         "    try:\n"
         "        res = ude.time_series_cv(\n"
         "            data,\n"
         "            model_class=ude.NODE,\n"
         "            model_kwargs={'hidden_units': best_h, 'hidden_layers': 2},\n"
         "            train_kwargs={'loss': loss_fn, 'epochs': 50, 'verbose': False},\n"
         "            n_splits=4, metric='rmse',\n"
         "        )\n"
         "        loss_scores[loss_fn] = np.mean([r['val_score'] for r in res])\n"
         "    except ImportError:\n"
         "        print(f'{loss_fn}: torchdiffeq not installed, skipping')\n"
         "        continue\n"
         "    print(f'{loss_fn:25s}  mean RMSE={loss_scores[loss_fn]:.4f}')"),

    md("## 7 — Per-State Scores (Multi-State)\n\n"
       "For multi-state systems (e.g. Lotka-Volterra), `state_scores` tells you which\n"
       "state variable is harder to predict."),

    code("def lv_rhs(t, u, alpha=1.0, beta=0.1, delta=0.075, gamma=1.5):\n"
         "    x, y = u\n"
         "    return [alpha*x - beta*x*y, delta*x*y - gamma*y]\n\n"
         "lv_sol = solve_ivp(lv_rhs, [0, 15], [10.0, 5.0],\n"
         "                   t_eval=np.linspace(0, 15, 120), method='RK45')\n"
         "lv_data = pd.DataFrame({'time': lv_sol.t,\n"
         "                        'prey': lv_sol.y[0] + 0.5*np.random.randn(120),\n"
         "                        'predator': lv_sol.y[1] + 0.5*np.random.randn(120)})\n\n"
         "lv_res = ude.time_series_cv(\n"
         "    lv_data,\n"
         "    model_class=ude.NODE,\n"
         "    model_kwargs={'hidden_units': 32, 'hidden_layers': 2},\n"
         "    train_kwargs={'loss': 'derivative_matching', 'epochs': 50, 'verbose': False},\n"
         "    n_splits=3, metric='rmse',\n"
         ")\n\n"
         "for r in lv_res:\n"
         "    ss = r['state_scores']\n"
         "    print(f\"fold {r['fold']}  prey={ss['prey']:.3f}  predator={ss['predator']:.3f}  mean={r['val_score']:.3f}\")"),

    md("## 8 — Final Model: Retrain on Full Data"),

    code("final_model = ude.NODE(data, hidden_units=best_h, hidden_layers=2)\n"
         "final_model.train(loss='derivative_matching', epochs=100, verbose=False)\n\n"
         "fc_final = final_model.forecast(steps=50)\n\n"
         "fig, ax = plt.subplots(figsize=(10, 4))\n"
         "ax.plot(data['time'], data['x'], 'b.', ms=4, label='data')\n"
         "ax.plot(fc_final['time'], fc_final['x'], 'r-', lw=2, label=f'forecast (h={best_h})')\n"
         "ax.set_xlabel('time'); ax.legend(); ax.set_title('Final model forecast')\n"
         "plt.tight_layout(); plt.show()"),

    md("## Key Takeaways\n\n"
       "- **Expanding-window CV** respects time order — validation is always in the future.\n"
       "- **`time_series_cv`** handles the full fold loop: instantiate → train → forecast → score.\n"
       "- **`state_scores`** per fold reveals which state variables are hardest to predict.\n"
       "- **Grid search** over `hidden_units` with CV avoids over-fitting hyperparameters to a\n"
       "  single test window.\n"
       "- Retrain the final model on the **full dataset** after hyperparameter selection."),
])

# ===========================================================================
# Notebook 06 — Parameter Recovery
# ===========================================================================

nb06 = nb("06_parameter_recovery", [
    md("# Notebook 06 — Parameter Recovery with CustomDerivatives\n\n"
       "One of the most powerful applications of UDEs is **recovering mechanistic parameters\n"
       "from noisy data** when the structure of the dynamics is partially known. This notebook\n"
       "demonstrates the full scientific workflow:\n\n"
       "1. Generate data from a known system (SIR epidemic model)\n"
       "2. Inject partial knowledge — recovery rate structure is known, transmission is not\n"
       "3. Train `CustomDerivatives` to learn the unknown transmission while recovering `γ`\n"
       "4. Inspect learned parameters and the neural transmission function\n"
       "5. Forecast beyond the training window"),

    code("import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import torch\n"
         "from scipy.integrate import solve_ivp\n"
         "import pyUDE as ude\n\n"
         "np.random.seed(3)\n"
         "torch.manual_seed(3)"),

    md("## 1 — Simulate the SIR Model\n\n"
       "The SIR (Susceptible–Infectious–Recovered) model:\n"
       "```\n"
       "dS/dt = -β * S * I / N\n"
       "dI/dt =  β * S * I / N  -  γ * I\n"
       "dR/dt =  γ * I\n"
       "```\n"
       "True parameters: `β = 0.3`, `γ = 0.1`, `N = 1000`."),

    code("N     = 1000.0\n"
         "BETA  = 0.3\n"
         "GAMMA = 0.1\n"
         "S0, I0, R0 = 990.0, 10.0, 0.0\n\n"
         "def sir(t, u):\n"
         "    S, I, R = u\n"
         "    dS = -BETA * S * I / N\n"
         "    dI =  BETA * S * I / N - GAMMA * I\n"
         "    dR =  GAMMA * I\n"
         "    return [dS, dI, dR]\n\n"
         "t_span  = [0, 80]\n"
         "t_all = np.linspace(0, 80, 110)\n\n"
         "sol_full = solve_ivp(sir, [0, 80], [S0, I0, R0],\n"
         "                     t_eval=t_all, method='RK45')\n\n"
         "# Add 2% observation noise\n"
         "noise = 0.02\n"
         "n_all = len(sol_full.t)\n"
         "cols  = ['S', 'I', 'R']\n"
         "data_all = pd.DataFrame({'time': sol_full.t,\n"
         "    'S': sol_full.y[0] * (1 + noise * np.random.randn(n_all)),\n"
         "    'I': sol_full.y[1] * (1 + noise * np.random.randn(n_all)),\n"
         "    'R': sol_full.y[2] * (1 + noise * np.random.randn(n_all)),\n"
         "})\n"
         "train_data = data_all.iloc[:80].reset_index(drop=True)\n"
         "test_data  = data_all.iloc[80:].reset_index(drop=True)\n\n"
         "fig, axes = plt.subplots(1, 3, figsize=(13, 3))\n"
         "for ax, col in zip(axes, cols):\n"
         "    ax.plot(train_data['time'], train_data[col], 'b.', ms=3, label='train')\n"
         "    ax.plot(test_data['time'],  test_data[col],  'r.', ms=3, label='test')\n"
         "    ax.set_title(col); ax.legend(fontsize=7)\n"
         "plt.suptitle('SIR epidemic data (2% noise)'); plt.tight_layout(); plt.show()"),

    md("## 2 — Why NODE Cannot Recover Parameters\n\n"
       "A plain `NODE` can fit the data, but it has no `get_params()` — the dynamics are\n"
       "entirely encoded in opaque network weights with no interpretable structure."),

    code("node = ude.NODE(train_data, hidden_units=32)\n"
         "node.train(loss='derivative_matching', epochs=150, verbose=False)\n"
         "print('NODE is_trained:', node.is_trained)\n"
         "print('NODE has get_params:', hasattr(node, 'get_params'))\n"
         "fc_node = node.forecast(steps=len(test_data))\n"
         "rmse_node = ude.rmse(test_data[cols].reset_index(drop=True),\n"
         "                     fc_node[cols].reset_index(drop=True))['mean']\n"
         "print(f'NODE test RMSE: {rmse_node:.2f}')"),

    md("## 3 — CustomDerivatives: Inject Partial Knowledge\n\n"
       "We know the **recovery structure** (`γ * I` removes individuals from I and adds them\n"
       "to R). We do **not** know the transmission (`β * S * I / N`). The NN learns it."),

    code("def sir_known(u, p, t):\n"
         "    \"\"\"Known terms: recovery from I, gain in R. Transmission learned by NN.\"\"\"\n"
         "    S, I, R = u[0], u[1], u[2]\n"
         "    gamma = p['gamma']\n"
         "    return torch.stack([\n"
         "        torch.zeros_like(S),   # S: transmission unknown — NN will fill this\n"
         "        -gamma * I,            # I: recovery (known)\n"
         "         gamma * I,            # R: gain from recovery (known)\n"
         "    ])\n\n"
         "model = ude.CustomDerivatives(\n"
         "    train_data, sir_known,\n"
         "    init_params={'gamma': 0.05},\n"
         "    param_bounds={'gamma': (0.0, 1.0)},\n"
         "    hidden_units=32,\n"
         ")\n"
         "print(model)"),

    md("## 4 — Multi-Stage Training\n\n"
       "Stage 1: derivative matching (fast, no integration). Stage 2: simulation loss\n"
       "(accurate, full ODE integration) with early stopping on training loss."),

    code("# Stage 1 — fast warm-up\n"
         "model.train(loss='derivative_matching', epochs=200,\n"
         "            learning_rate=1e-3, verbose=False)\n\n"
         "# Stage 2 — accurate fine-tuning (requires torchdiffeq)\n"
         "try:\n"
         "    model.train(loss='simulation', epochs=200,\n"
         "                learning_rate=3e-4, patience=50, verbose=False)\n"
         "    print('Stage 2 (simulation) complete.')\n"
         "except ImportError:\n"
         "    print('torchdiffeq not installed — skipping simulation stage.')\n\n"
         "print(f'Total epochs trained: {len(model.train_history_[\"train_loss\"])}')"),

    md("## 5 — Parameter Recovery"),

    code("params = model.get_params()\n"
         "print(f'Recovered γ = {params[\"gamma\"]:.4f}   (true γ = {GAMMA})')\n"
         "print(f'Relative error: {abs(params[\"gamma\"] - GAMMA) / GAMMA * 100:.1f}%')\n\n"
         "# Plot training history with stage boundary\n"
         "history = model.train_history_\n"
         "n_dm = 200  # epochs in stage 1\n\n"
         "fig, ax = plt.subplots(figsize=(9, 4))\n"
         "ax.semilogy(range(1, len(history['train_loss'])+1), history['train_loss'], lw=1.5)\n"
         "ax.axvline(n_dm, color='k', ls='--', label='DM → simulation')\n"
         "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (log scale)')\n"
         "ax.legend(); ax.set_title('Training loss (DM warm-up then simulation)')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 6 — Inspect the Learned Transmission Term\n\n"
       "Extract `get_right_hand_side()` and evaluate the NN contribution on a grid of\n"
       "`(S, I)` values to see what transmission function was learned."),

    code("rhs = model.get_right_hand_side()  # f(u, t) -> du, numpy-compatible\n\n"
         "# Evaluate NN contribution to dS/dt on an (S, I) grid\n"
         "S_vals = np.linspace(0, 1000, 40)\n"
         "I_vals = np.linspace(0, 500, 40)\n"
         "SS, II = np.meshgrid(S_vals, I_vals)\n\n"
         "# Theoretical: dS/dt transmission = -beta * S * I / N\n"
         "true_trans = -BETA * SS * II / N\n\n"
         "learned_trans = np.zeros_like(SS)\n"
         "for i in range(40):\n"
         "    for j in range(40):\n"
         "        u = np.array([SS[i,j], II[i,j], N - SS[i,j] - II[i,j]])\n"
         "        du = rhs(u, 0.0)\n"
         "        learned_trans[i, j] = du[0]  # dS/dt component\n\n"
         "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
         "for ax, Z, title in zip(axes,\n"
         "                        [true_trans, learned_trans],\n"
         "                        ['True: -βSI/N', 'Learned by NN']):\n"
         "    im = ax.pcolormesh(SS, II, Z, cmap='RdBu_r',\n"
         "                       vmin=true_trans.min(), vmax=0)\n"
         "    plt.colorbar(im, ax=ax)\n"
         "    ax.set_xlabel('S'); ax.set_ylabel('I')\n"
         "    ax.set_title(title)\n"
         "plt.suptitle('Transmission term dS/dt'); plt.tight_layout(); plt.show()"),

    md("## 7 — Forecast and Evaluation"),

    code("fc = model.forecast(steps=len(test_data))\n"
         "min_len = min(len(fc), len(test_data))\n"
         "obs_test = test_data[cols].iloc[:min_len].reset_index(drop=True)\n"
         "pred_test = fc[cols].iloc[:min_len].reset_index(drop=True)\n\n"
         "r2 = ude.r2_score(obs_test, pred_test)\n"
         "rmse_cd = ude.rmse(obs_test, pred_test)\n"
         "print('R² per state:  ', r2)\n"
         "print('RMSE per state:', rmse_cd)\n\n"
         "fig, axes = plt.subplots(1, 3, figsize=(13, 3))\n"
         "for ax, col in zip(axes, cols):\n"
         "    ax.plot(train_data['time'], train_data[col], 'b.', ms=3, label='train')\n"
         "    ax.plot(test_data['time'],  test_data[col],  'r.', ms=3, label='true test')\n"
         "    ax.plot(fc['time'], fc[col], 'k-', lw=2, label='forecast')\n"
         "    ax.axvline(train_data['time'].iloc[-1], color='k', ls='--', lw=1)\n"
         "    ax.set_title(col); ax.legend(fontsize=7)\n"
         "plt.suptitle('SIR forecast vs observations'); plt.tight_layout(); plt.show()"),

    md("## 8 — Effect of `param_bounds`"),

    code("model_no_bounds = ude.CustomDerivatives(\n"
         "    train_data, sir_known,\n"
         "    init_params={'gamma': 0.05},   # no bounds\n"
         "    hidden_units=32,\n"
         ")\n"
         "model_no_bounds.train(loss='derivative_matching', epochs=200, verbose=False)\n"
         "gamma_nb = model_no_bounds.get_params()['gamma']\n"
         "gamma_b  = model.get_params()['gamma']\n\n"
         "print(f'True γ:         {GAMMA}')\n"
         "print(f'Recovered (bounded):   {gamma_b:.4f}')\n"
         "print(f'Recovered (unbounded): {gamma_nb:.4f}  '\n"
         "      f'({\"NEGATIVE — physically invalid!\" if gamma_nb < 0 else \"OK\"})')"),

    md("## Key Takeaways\n\n"
       "- `CustomDerivatives` separates **known mechanistic structure** from **unknown residuals**.\n"
       "- `get_params()` returns learned parameter values as plain floats — directly interpretable.\n"
       "- `param_bounds` prevents the optimizer from pushing parameters into physically impossible\n"
       "  ranges (e.g. negative rates).\n"
       "- `get_right_hand_side()` lets you inspect the learned NN term as a function over state space.\n"
       "- Multi-stage training (DM warm-up → simulation fine-tune) is the recommended workflow\n"
       "  for parameter recovery tasks."),
])

# ===========================================================================
# Notebook 07 — Custom Networks & Model Persistence
# ===========================================================================

nb07 = nb("07_custom_networks", [
    md("# Notebook 07 — Custom Networks & Model Persistence\n\n"
       "This notebook covers two advanced topics:\n\n"
       "1. **Custom `nn.Module` architectures** — when to replace the default MLP\n"
       "   and how to pass your own network to any model class.\n"
       "2. **Save/load and multi-stage training** — checkpointing, resuming, and\n"
       "   a production-ready end-to-end workflow template."),

    code("import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import torch\n"
         "import torch.nn as nn\n"
         "from scipy.integrate import solve_ivp\n"
         "import pyUDE as ude\n\n"
         "np.random.seed(5)\n"
         "torch.manual_seed(5)"),

    md("## 1 — Generate Data: Van der Pol Oscillator\n\n"
       "The Van der Pol oscillator has stiff nonlinear dynamics:\n"
       "```\n"
       "dx/dt = y\n"
       "dy/dt = μ(1 - x²)y - x\n"
       "```\n"
       "with `μ = 2.0`. The limit cycle and varying time scales make this a challenging\n"
       "system for a small default MLP."),

    code("mu = 2.0\n\n"
         "def van_der_pol(t, u):\n"
         "    x, y = u\n"
         "    return [y, mu*(1 - x**2)*y - x]\n\n"
         "sol = solve_ivp(van_der_pol, [0, 20], [2.0, 0.0],\n"
         "                t_eval=np.linspace(0, 20, 200),\n"
         "                method='Radau')  # stiff solver\n\n"
         "data = pd.DataFrame({'time': sol.t, 'x': sol.y[0], 'y': sol.y[1]})\n"
         "train_data, test_data = ude.train_test_split(data, test_fraction=0.2)\n\n"
         "fig, ax = plt.subplots(figsize=(10, 3))\n"
         "ax.plot(data['time'], data['x'], 'b-', lw=1.5, label='x')\n"
         "ax.plot(data['time'], data['y'], 'r-', lw=1.5, label='y', alpha=0.7)\n"
         "ax.axvline(train_data['time'].iloc[-1], color='k', ls='--', label='test split')\n"
         "ax.legend(); ax.set_xlabel('time'); ax.set_title('Van der Pol oscillator')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 2 — Default MLP Baseline"),

    code("model_default = ude.NODE(train_data, hidden_units=64, hidden_layers=2)\n"
         "model_default.train(loss='derivative_matching', epochs=150, verbose=False)\n\n"
         "fc_default = model_default.forecast(steps=len(test_data))\n"
         "min_len = min(len(fc_default), len(test_data))\n"
         "state_cols = ['x', 'y']\n"
         "rmse_default = ude.rmse(\n"
         "    test_data[state_cols].iloc[:min_len].reset_index(drop=True),\n"
         "    fc_default[state_cols].iloc[:min_len].reset_index(drop=True)\n"
         ")['mean']\n"
         "print(f'Default MLP test RMSE: {rmse_default:.4f}')"),

    md("## 3 — Residual Network\n\n"
       "A residual (skip) connection adds the input back to the output:\n"
       "`output = f(u) + u`. This helps when dynamics are near-identity — small corrections\n"
       "to the current state — because the network only needs to learn the deviation."),

    code("class ResidualMLP(nn.Module):\n"
         "    def __init__(self, dim: int, hidden: int = 64):\n"
         "        super().__init__()\n"
         "        self.fc1 = nn.Linear(dim, hidden)\n"
         "        self.fc2 = nn.Linear(hidden, hidden)\n"
         "        self.fc3 = nn.Linear(hidden, dim)\n\n"
         "    def forward(self, u: torch.Tensor) -> torch.Tensor:\n"
         "        h = torch.tanh(self.fc1(u))\n"
         "        h = torch.tanh(self.fc2(h))\n"
         "        return self.fc3(h) + u   # skip connection\n\n"
         "# Pass via network= parameter — set dtype to float64 to match training data\n"
         "model_res = ude.NODE(train_data, network=ResidualMLP(2, hidden=64).double())\n"
         "model_res.train(loss='derivative_matching', epochs=150, verbose=False)\n\n"
         "fc_res = model_res.forecast(steps=len(test_data))\n"
         "rmse_res = ude.rmse(\n"
         "    test_data[state_cols].iloc[:min_len].reset_index(drop=True),\n"
         "    fc_res[state_cols].iloc[:min_len].reset_index(drop=True)\n"
         ")['mean']\n"
         "print(f'Residual MLP test RMSE: {rmse_res:.4f}')\n"
         "print(f'Improvement: {(rmse_default - rmse_res) / rmse_default * 100:.1f}%')"),

    code("fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
         "for ax, col in zip(axes, state_cols):\n"
         "    ax.plot(data['time'], data[col], 'k.', ms=3, alpha=0.4, label='observed')\n"
         "    ax.plot(fc_default['time'], fc_default[col], 'b-', lw=2,\n"
         "            label=f'default MLP (RMSE={rmse_default:.3f})')\n"
         "    ax.plot(fc_res['time'], fc_res[col], 'r-', lw=2,\n"
         "            label=f'residual MLP (RMSE={rmse_res:.3f})')\n"
         "    ax.axvline(train_data['time'].iloc[-1], color='k', ls='--', lw=1)\n"
         "    ax.set_title(col); ax.legend(fontsize=8)\n"
         "plt.suptitle('Default vs Residual Network'); plt.tight_layout(); plt.show()"),

    md("## 4 — When to Use Custom Networks\n\n"
       "| Scenario | Architecture suggestion |\n"
       "|----------|------------------------|\n"
       "| Dynamics are near-identity | Residual / skip connections |\n"
       "| Default MLP underfits | Increase `hidden_units` or `hidden_layers` first |\n"
       "| Known conserved quantities | Structured network with conservation built in |\n"
       "| Fast-slow dynamics | Larger network or implicit integration (Julia) |\n"
       "| General unknown dynamics | Default MLP (2 layers, 32 units) is a good start |\n\n"
       "> **Important:** any custom network must call `.double()` to match pyUDE's float64 tensors."),

    md("## 5 — `save()` and `load_weights()`"),

    code("import tempfile, os\n\n"
         "# Save to a temporary file\n"
         "tmpfile = os.path.join(tempfile.gettempdir(), 'van_der_pol_residual.pt')\n"
         "model_res.save(tmpfile)\n"
         "print(f'Saved to: {tmpfile}')\n\n"
         "# Re-instantiate with the same architecture and load weights\n"
         "model_reloaded = ude.NODE(train_data, network=ResidualMLP(2, hidden=64).double())\n"
         "model_reloaded.load_weights(tmpfile)\n"
         "print('Loaded.  is_trained:', model_reloaded.is_trained)\n\n"
         "# Verify forecasts are identical\n"
         "fc_reloaded = model_reloaded.forecast(steps=len(test_data))\n"
         "max_diff = (fc_res[state_cols].values - fc_reloaded[state_cols].values).max()\n"
         "print(f'Max abs difference between saved and reloaded forecast: {max_diff:.2e}')"),

    md("## 6 — Multi-Stage Training with `train_history_`\n\n"
       "Each call to `train()` continues from the current weights. The `train_history_`\n"
       "attribute accumulates across all calls, so you can plot the full training trajectory."),

    code("model_staged = ude.NODE(train_data, hidden_units=64, hidden_layers=3)\n\n"
         "# Stage 1: fast warm-up with high learning rate\n"
         "model_staged.train(loss='derivative_matching', epochs=100,\n"
         "                   learning_rate=1e-3, verbose=False)\n"
         "phase1_end = len(model_staged.train_history_['train_loss'])\n\n"
         "# Stage 2: refine with lower learning rate\n"
         "model_staged.train(loss='derivative_matching', epochs=100,\n"
         "                   learning_rate=3e-4, verbose=False)\n"
         "phase2_end = len(model_staged.train_history_['train_loss'])\n\n"
         "# Stage 3: fine-tune with weight decay for regularization\n"
         "model_staged.train(loss='derivative_matching', epochs=100,\n"
         "                   learning_rate=1e-4, weight_decay=1e-4, verbose=False)\n\n"
         "print(f'Total epochs: {len(model_staged.train_history_[\"train_loss\"])}')"),

    code("history = model_staged.train_history_['train_loss']\n"
         "epochs  = range(1, len(history) + 1)\n"
         "phases  = [(1, phase1_end, 'warm-up (lr=1e-3)'),\n"
         "           (phase1_end+1, phase2_end, 'refine (lr=3e-4)'),\n"
         "           (phase2_end+1, len(history), 'fine-tune (lr=1e-4, wd)')]\n\n"
         "fig, ax = plt.subplots(figsize=(10, 4))\n"
         "ax.semilogy(epochs, history, lw=1.5, color='steelblue')\n"
         "colors = ['#ff9999', '#99cc99', '#9999ff']\n"
         "for (start, end, label), c in zip(phases, colors):\n"
         "    if end >= start:\n"
         "        ax.axvspan(start, end, alpha=0.25, color=c, label=label)\n"
         "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (log scale)')\n"
         "ax.legend(fontsize=8); ax.set_title('Multi-stage training history')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 7 — CustomDifferences Save/Load"),

    code("def logistic_map(u, p, t):\n"
         "    return torch.stack([p['r'] * u[0] * (1 - u[0] / 10.0)])\n\n"
         "t_map = np.arange(50)\n"
         "x_map = np.zeros(50); x_map[0] = 0.5\n"
         "r_true = 0.9\n"
         "for i in range(1, 50):\n"
         "    x_map[i] = r_true * x_map[i-1] * (1 - x_map[i-1] / 10.0) + 0.01*np.random.randn()\n\n"
         "map_data = pd.DataFrame({'time': t_map.astype(float), 'x': x_map})\n"
         "cd_model = ude.CustomDifferences(map_data, logistic_map, init_params={'r': 0.5})\n"
         "cd_model.train(epochs=200, verbose=False)\n"
         "print('Learned r:', cd_model.get_params())\n\n"
         "# Save and reload\n"
         "cd_path = os.path.join(tempfile.gettempdir(), 'logistic_diff.pt')\n"
         "cd_model.save(cd_path)\n"
         "cd_reload = ude.CustomDifferences(map_data, logistic_map, init_params={'r': 0.5})\n"
         "cd_reload.load_weights(cd_path)\n"
         "print('Reloaded r:', cd_reload.get_params())"),

    md("## 8 — Production Workflow Template\n\n"
       "A clean end-to-end pattern you can copy-paste as a starting point for new projects."),

    code("import tempfile, os\n"
         "import numpy as np, pandas as pd\n"
         "import torch.nn as nn\n"
         "import pyUDE as ude\n\n"
         "# 1. Load or generate data\n"
         "# data = pd.read_csv('my_data.csv')\n"
         "# Using synthetic data for this demo:\n"
         "t = np.linspace(0, 10, 100)\n"
         "demo_data = pd.DataFrame({'time': t, 'x': np.sin(t), 'y': np.cos(t)})\n\n"
         "# 2. Temporal split\n"
         "train_df, test_df = ude.train_test_split(demo_data, test_fraction=0.2)\n\n"
         "# 3. Build model (default MLP or custom network)\n"
         "final = ude.NODE(train_df, hidden_units=32, hidden_layers=2)\n\n"
         "# 4. Multi-stage training with validation tracking\n"
         "final.train(loss='derivative_matching', epochs=100,\n"
         "            val_data=test_df, val_interval=10,\n"
         "            learning_rate=1e-3, verbose=False)\n\n"
         "# 5. Evaluate on test set\n"
         "fc = final.forecast(steps=len(test_df))\n"
         "min_l = min(len(fc), len(test_df))\n"
         "scores = ude.rmse(test_df[['x','y']].iloc[:min_l].reset_index(drop=True),\n"
         "                  fc[['x','y']].iloc[:min_l].reset_index(drop=True))\n"
         "print('Test RMSE:', scores)\n\n"
         "# 6. Save checkpoint\n"
         "ckpt = os.path.join(tempfile.gettempdir(), 'final_model.pt')\n"
         "final.save(ckpt)\n"
         "print(f'Model saved to {ckpt}')\n\n"
         "# 7. Load and forecast in new session\n"
         "loaded = ude.NODE(train_df, hidden_units=32, hidden_layers=2)\n"
         "loaded.load_weights(ckpt)\n"
         "future = loaded.forecast(steps=50)\n"
         "print('Future forecast shape:', future.shape)"),

    md("## Key Takeaways\n\n"
       "- Pass any `nn.Module` via `network=` to override the default MLP — always call\n"
       "  `.double()` on your network to match pyUDE's float64 tensors.\n"
       "- **Residual connections** help when dynamics are near-identity (small corrections).\n"
       "- `save()` stores the network state dict and training metadata; `load_weights()` restores\n"
       "  it — the **model architecture must match** at load time.\n"
       "- `train_history_` accumulates across all `train()` calls — use it to plot the full\n"
       "  multi-stage learning curve.\n"
       "- `CustomDifferences` has its own checkpoint format (no `_ode_func`); use its\n"
       "  own `save()` / `load_weights()` overrides."),
])

# ===========================================================================
# Notebook 08 — Julia Backend
# ===========================================================================

nb08 = nb("08_julia_backend", [
    md("# Notebook 08 — Julia Backend\n\n"
       "pyUDE provides an optional Julia backend (`JuliaNODE`, `JuliaCustomDerivatives`,\n"
       "`JuliaCustomDifferences`) that delegates ODE integration and network training to\n"
       "[UniversalDiffEq.jl](https://jack-h-buckner.github.io/UniversalDiffEq.jl/dev/).\n\n"
       "**Choose Julia when you need:**\n"
       "- Stiff ODE solvers (`Rodas5`, `QNDF`) from DifferentialEquations.jl\n"
       "- High-order, high-accuracy solvers (`Vern9`)\n"
       "- Tight integration with the Julia scientific ecosystem\n\n"
       "**Choose PyTorch when you need:**\n"
       "- GPU acceleration, custom architectures, dropout, `param_bounds`, `save()`/`load_weights()`\n\n"
       "> All Julia cells are wrapped in `if JULIA_AVAILABLE:` guards so this notebook runs\n"
       "> without error on machines where Julia is not installed."),

    md("## Installation\n\n"
       "```bash\n"
       "# 1. Install Julia >= 1.10 from https://julialang.org/downloads\n"
       "# 2. Install pyUDE with the Julia extra:\n"
       "pip install 'pyUDE[julia]'\n"
       "# 3. First run: Julia auto-downloads and compiles UniversalDiffEq.jl (~2–5 min)\n"
       "```\n\n"
       "> **First-run JIT warm-up:** The first call to `model.train()` starts Julia and\n"
       "> compiles the package. Expect ~60 s on first use; subsequent calls are fast (~1 s)."),

    code("import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "from scipy.integrate import solve_ivp\n"
         "import pyUDE as ude\n\n"
         "np.random.seed(11)\n\n"
         "try:\n"
         "    import juliacall  # must be importable for Julia to work\n"
         "    from pyUDE import JuliaNODE, JuliaCustomDerivatives, JuliaCustomDifferences\n"
         "    JULIA_AVAILABLE = True\n"
         "    print('Julia backend available.')\n"
         "except ImportError:\n"
         "    JULIA_AVAILABLE = False\n"
         "    JuliaNODE = JuliaCustomDerivatives = JuliaCustomDifferences = None\n"
         "    print('Julia backend not available — install with: pip install \"pyUDE[julia]\"')"),

    md("## 1 — Generate Data: Lotka-Volterra"),

    code("def lv_rhs(t, u):\n"
         "    x, y = u\n"
         "    return [x - 0.1*x*y, -1.5*y + 0.075*x*y]\n\n"
         "sol = solve_ivp(lv_rhs, [0, 15], [10.0, 5.0],\n"
         "                t_eval=np.linspace(0, 15, 80), method='RK45')\n"
         "lv_data = pd.DataFrame({'time': sol.t, 'prey': sol.y[0], 'predator': sol.y[1]})\n\n"
         "fig, ax = plt.subplots(figsize=(9, 3))\n"
         "ax.plot(sol.t, sol.y[0], label='prey')\n"
         "ax.plot(sol.t, sol.y[1], label='predator')\n"
         "ax.legend(); ax.set_xlabel('time'); ax.set_title('Lotka-Volterra')\n"
         "plt.tight_layout(); plt.show()"),

    md("## 2 — `JuliaNODE` — Basic Usage\n\n"
       "The constructor signature matches `NODE` but there is no `device=` or `network=`\n"
       "parameter. The solver is set at construction time (not training time)."),

    code("if JULIA_AVAILABLE:\n"
         "    model_jnode = JuliaNODE(\n"
         "        lv_data,\n"
         "        hidden_units=32,\n"
         "        hidden_layers=2,\n"
         "        solver='Tsit5',   # explicit 5th-order, good for non-stiff\n"
         "    )\n"
         "    print(model_jnode)\n"
         "    print('Julia starts lazily — training cell will trigger JIT compilation...')\n"
         "else:\n"
         "    print('Skipping JuliaNODE — Julia not installed.')"),

    code("if JULIA_AVAILABLE:\n"
         "    model_jnode.train(epochs=300, learning_rate=1e-3, verbose=True)\n"
         "    print('is_trained:', model_jnode.is_trained)\n"
         "    fc_jnode = model_jnode.forecast(steps=30)\n"
         "    print('Forecast shape:', fc_jnode.shape, fc_jnode.head(3))"),

    md("## 3 — Solver Reference\n\n"
       "| Solver | Type | Use case |\n"
       "|--------|------|----------|\n"
       "| `\"Tsit5\"` | Explicit 5th-order | Default, non-stiff |\n"
       "| `\"Vern9\"` | Explicit 9th-order | High accuracy, smooth |\n"
       "| `\"Rodas5\"` | Implicit 5th-order | **Stiff** (fast–slow dynamics) |\n"
       "| `\"QNDF\"` | Implicit variable-order | Stiff, like MATLAB `ode15s` |\n"
       "| `\"AutoTsit5(Rosenbrock23())\"` | Auto-switching | Unknown stiffness |\n\n"
       "Pass the solver string at **construction time**, not training time."),

    md("## 4 — Stiff System: Van der Pol with `Rodas5`\n\n"
       "`Tsit5` (explicit) can fail or diverge on stiff systems because the step-size\n"
       "constraint becomes very restrictive. An implicit solver like `Rodas5` handles\n"
       "stiff problems efficiently."),

    code("def vdp(t, u, mu=2.0):\n"
         "    x, y = u\n"
         "    return [y, mu*(1 - x**2)*y - x]\n\n"
         "vdp_sol = solve_ivp(vdp, [0, 20], [2.0, 0.0],\n"
         "                    t_eval=np.linspace(0, 20, 120), method='Radau')\n"
         "vdp_data = pd.DataFrame({'time': vdp_sol.t, 'x': vdp_sol.y[0], 'y': vdp_sol.y[1]})\n\n"
         "if JULIA_AVAILABLE:\n"
         "    # Stiff solver\n"
         "    model_stiff = JuliaNODE(vdp_data, hidden_units=32, solver='Rodas5')\n"
         "    model_stiff.train(epochs=300, learning_rate=1e-3, verbose=False)\n"
         "    fc_stiff = model_stiff.forecast(steps=20)\n\n"
         "    fig, ax = plt.subplots(figsize=(10, 3))\n"
         "    ax.plot(vdp_data['time'], vdp_data['x'], 'k.', ms=3, alpha=0.5, label='data')\n"
         "    ax.plot(fc_stiff['time'], fc_stiff['x'], 'r-', lw=2, label='forecast (Rodas5)')\n"
         "    ax.axvline(vdp_data['time'].iloc[-1], color='k', ls='--')\n"
         "    ax.legend(); ax.set_title('Van der Pol — JuliaNODE with Rodas5')\n"
         "    plt.tight_layout(); plt.show()\n"
         "else:\n"
         "    print('Skipping — Julia not installed.')"),

    md("## 5 — `JuliaCustomDerivatives` — Signature Difference\n\n"
       "The `known_dynamics` function for the Julia backend receives **plain Python lists\n"
       "and floats**, not PyTorch tensors. This is because the function is called from\n"
       "Julia via PythonCall.jl during ODE integration.\n\n"
       "```python\n"
       "# PyTorch backend — tensors\n"
       "def known_pt(u, p, t):\n"
       "    return torch.stack([p['alpha'] * u[0], -p['delta'] * u[1]])\n\n"
       "# Julia backend — plain lists/floats\n"
       "def known_jl(u, p, t):\n"
       "    return [p['alpha'] * u[0], -p['delta'] * u[1]]\n"
       "```"),

    code("# Julia backend known_dynamics: u is list, p values are floats, t is float\n"
         "def lv_known_julia(u, p, t):\n"
         "    prey, pred = u[0], u[1]\n"
         "    return [p['alpha'] * prey, -p['delta'] * pred]\n\n"
         "if JULIA_AVAILABLE:\n"
         "    model_jcd = JuliaCustomDerivatives(\n"
         "        lv_data, lv_known_julia,\n"
         "        init_params={'alpha': 0.8, 'delta': 1.2},\n"
         "        hidden_units=32,\n"
         "        solver='Tsit5',\n"
         "    )\n"
         "    model_jcd.train(epochs=400, learning_rate=1e-3, verbose=False)\n"
         "    print('Recovered params:', model_jcd.get_params())\n"
         "else:\n"
         "    print('Skipping — Julia not installed.')"),

    md("## 6 — `JuliaCustomDifferences`"),

    code("t_map = np.arange(60, dtype=float)\n"
         "x_map = np.zeros(60); x_map[0] = 3.6\n"
         "for i in range(1, 60):\n"
         "    x_map[i] = 3.6 * x_map[i-1] * (1 - x_map[i-1] / 10) + 0.05*np.random.randn()\n"
         "map_data = pd.DataFrame({'time': t_map, 'x': x_map})\n\n"
         "def logistic_known_jl(u, p, t):\n"
         "    \"\"\"Julia backend: u and p are plain Python types.\"\"\"\n"
         "    return [p['r'] * u[0] * (1 - u[0] / 10.0)]\n\n"
         "if JULIA_AVAILABLE:\n"
         "    model_jdiff = JuliaCustomDifferences(\n"
         "        map_data, logistic_known_jl, init_params={'r': 3.0}\n"
         "    )\n"
         "    model_jdiff.train(epochs=300, learning_rate=1e-3, verbose=False)\n"
         "    print('Recovered r:', model_jdiff.get_params())\n"
         "else:\n"
         "    print('Skipping — Julia not installed.')"),

    md("## 7 — `get_right_hand_side()` with scipy\n\n"
       "The Julia-trained model's RHS is callable from Python tools via PythonCall.jl."),

    code("if JULIA_AVAILABLE and model_jnode.is_trained:\n"
         "    from scipy.integrate import odeint\n\n"
         "    rhs = model_jnode.get_right_hand_side()\n"
         "    u0 = lv_data[['prey', 'predator']].iloc[0].values\n"
         "    t_span = np.linspace(0, 20, 200)\n\n"
         "    sol_scipy = odeint(rhs, y0=u0, t=t_span)\n\n"
         "    fig, ax = plt.subplots(figsize=(9, 3))\n"
         "    ax.plot(lv_data['time'], lv_data['prey'], 'b.', ms=3, label='data (prey)')\n"
         "    ax.plot(t_span, sol_scipy[:, 0], 'b-', lw=2, label='scipy odeint (prey)')\n"
         "    ax.legend(); ax.set_title('Julia RHS integrated with scipy.odeint')\n"
         "    plt.tight_layout(); plt.show()\n"
         "else:\n"
         "    print('Skipping — Julia not installed or model not trained.')"),

    md("## 8 — PyTorch vs Julia Comparison"),

    code("import torch\n\n"
         "# PyTorch backend\n"
         "model_pt = ude.NODE(lv_data, hidden_units=32, hidden_layers=2)\n"
         "model_pt.train(loss='derivative_matching', epochs=150, verbose=False)\n"
         "fc_pt = model_pt.forecast(steps=30)\n\n"
         "_, test_lv = ude.train_test_split(lv_data, test_fraction=0.25)\n"
         "min_l = min(len(fc_pt), len(test_lv))\n"
         "state_cols = ['prey', 'predator']\n"
         "rmse_pt = ude.rmse(test_lv[state_cols].iloc[:min_l].reset_index(drop=True),\n"
         "                   fc_pt[state_cols].iloc[:min_l].reset_index(drop=True))['mean']\n"
         "print(f'PyTorch NODE test RMSE: {rmse_pt:.4f}')\n\n"
         "if JULIA_AVAILABLE and model_jnode.is_trained:\n"
         "    fc_jl = model_jnode.forecast(steps=30)\n"
         "    rmse_jl = ude.rmse(test_lv[state_cols].iloc[:min_l].reset_index(drop=True),\n"
         "                       fc_jl[state_cols].iloc[:min_l].reset_index(drop=True))['mean']\n"
         "    print(f'JuliaNODE test RMSE:   {rmse_jl:.4f}')\n"
         "else:\n"
         "    print('Julia comparison skipped — Julia not installed.')"),

    md("## 9 — Limitations\n\n"
       "| Feature | PyTorch backend | Julia backend |\n"
       "|---------|----------------|---------------|\n"
       "| GPU support (`device=`) | ✓ | — |\n"
       "| Custom `nn.Module` (`network=`) | ✓ | — |\n"
       "| `dropout` | ✓ | — |\n"
       "| `param_bounds` | ✓ | — |\n"
       "| `save()` / `load_weights()` | ✓ | — |\n"
       "| Derivative-matching loss | ✓ | — |\n"
       "| `val_data` / `train_history_` | ✓ | — |\n"
       "| Stiff solvers (`Rodas5`, `QNDF`) | limited | ✓ |\n"
       "| High-order solvers (`Vern9`) | — | ✓ |\n"
       "| DifferentialEquations.jl ecosystem | — | ✓ |"),

    md("## 10 — PythonCall Bridge Performance Note\n\n"
       "`JuliaCustomDerivatives` calls your Python `known_dynamics` function from Julia at\n"
       "**every ODE right-hand-side evaluation** — potentially millions of times per training\n"
       "run. Each call crosses the Python↔Julia boundary, adding ~1–5 µs overhead.\n\n"
       "**Recommendations:**\n"
       "- Keep `known_dynamics` simple (a few multiplications / additions)\n"
       "- Avoid Python loops or NumPy calls inside `known_dynamics`\n"
       "- If `known_dynamics` is complex, consider moving it to a Julia file using the\n"
       "  lower-level UniversalDiffEq.jl API directly"),

    md("## Key Takeaways\n\n"
       "- The Julia backend is **optional** (`pip install 'pyUDE[julia]'`) and starts lazily.\n"
       "- Use `JuliaNODE` / `JuliaCustomDerivatives` / `JuliaCustomDifferences` when stiff\n"
       "  solvers or high-order integration accuracy is needed.\n"
       "- `known_dynamics` for Julia uses **plain lists and floats** — no PyTorch operations.\n"
       "- `get_right_hand_side()` and `get_params()` work identically to the PyTorch backend.\n"
       "- For GPU, dropout, param_bounds, or save/load, use the PyTorch backend.\n"
       "- `if JULIA_AVAILABLE:` guards make notebooks portable to machines without Julia."),
])


# ===========================================================================
# Write all notebooks
# ===========================================================================

notebooks = {
    "03_validation_evaluation.ipynb": nb03,
    "04_regularization.ipynb": nb04,
    "05_cross_validation.ipynb": nb05,
    "06_parameter_recovery.ipynb": nb06,
    "07_custom_networks.ipynb": nb07,
    "08_julia_backend.ipynb": nb08,
}

print("Generating notebooks...")
for name, book in notebooks.items():
    write(name, book)
print("Done.")
