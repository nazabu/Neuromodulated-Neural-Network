"""
Neuromodulated Hebbian Learning
================================
Full implementation of "Final Project Report: Neuromodulated Hebbian Learning"
by Abubakr Nazriev (Mar 20, 2026).

This script implements and compares:
  Stage 1: Baseline MLP trained with backpropagation (Adam optimizer)
  Stage 2: Neuromodulated Hebbian Network (Dopamine + Acetylcholine signals)
  Ablation: Hebbian Network without ACh gating (demonstrates weight instability)

All neural network components are built from scratch using NumPy.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import gzip
import struct
import urllib.request
import time

# Reproducibility
np.random.seed(42)

# ====================================================================
# MNIST Data Loading
# ====================================================================


def load_mnist(data_dir="data"):
    """Download and load the MNIST dataset. Returns float32 arrays."""
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs(data_dir, exist_ok=True)

    def _download(filename):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"    Downloading {filename} ...")
            urllib.request.urlretrieve(base_url + filename, path)
        return path

    def _load_images(path):
        with gzip.open(path, "rb") as f:
            _magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0

    def _load_labels(path):
        with gzip.open(path, "rb") as f:
            _magic, _num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = _load_images(_download(files["train_images"]))
    y_train = _load_labels(_download(files["train_labels"]))
    X_test = _load_images(_download(files["test_images"]))
    y_test = _load_labels(_download(files["test_labels"]))
    return X_train, y_train, X_test, y_test


# ====================================================================
# Layer Classes
# ====================================================================


class Layer_Input:
    """Passthrough layer that stores raw input as its output."""

    def forward(self, inputs, training=False):
        self.output = inputs


class Layer_Dense:
    """Standard dense layer with backpropagation support (Stage 1)."""

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Hebbian:
    """
    Neuromodulated Hebbian layer (Stage 2).

    Replaces backpropagation with a three-factor local learning rule:
        delta_W = eta * ACh * DA * (x^T . y)
    where DA is a per-sample reward-prediction error and ACh is a scalar
    plasticity gate.
    """

    def __init__(self, n_inputs, n_neurons, learning_rate=0.01):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.learning_rate = learning_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def update_weights(self, dopamine_signal, ach_signal):
        """
        Neuromodulated Hebbian Update:
        delta_W = eta * ACh * DA * (x^T * y)

        Parameters
        ----------
        dopamine_signal : ndarray, shape (batch_size, 1) or (batch_size, n_out)
            Per-sample reward-prediction error.  For the hidden layer this is
            a scalar RPE broadcast across neurons; for the output layer this
            is the per-neuron error vector (target − prediction).
        ach_signal : float
            Scalar plasticity gate (0 → no update, 1 → full update).
        """
        batch_size = len(self.output)

        # x is presynaptic activation (inputs), y is postsynaptic (output)
        hebbian_trace = np.dot(self.inputs.T, self.output * dopamine_signal)
        hebbian_trace /= batch_size

        self.weights += self.learning_rate * ach_signal * hebbian_trace
        self.biases += (
            self.learning_rate
            * ach_signal
            * np.mean(self.output * dopamine_signal, axis=0, keepdims=True)
        )


# ====================================================================
# Activation Functions
# ====================================================================


class Activation_ReLU:
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs, training=False):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for idx, (single_out, single_dv) in enumerate(zip(self.output, dvalues)):
            single_out = single_out.reshape(-1, 1)
            jacobian = np.diagflat(single_out) - np.dot(single_out, single_out.T)
            self.dinputs[idx] = np.dot(jacobian, single_dv)


# ====================================================================
# Loss Functions
# ====================================================================


class Loss:
    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues / samples


class Activation_Softmax_Loss_CategoricalCrossentropy:
    """Combined Softmax + CCE backward for numerical stability."""

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


# ====================================================================
# Optimizer (Adam)
# ====================================================================


class Optimizer_Adam:
    def __init__(
        self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Momentum
        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        )
        weight_m_corr = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_m_corr = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Cache (RMSProp part)
        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        )
        layer.bias_cache = (
            self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        )
        weight_c_corr = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_c_corr = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Parameter update
        layer.weights += (
            -self.current_learning_rate
            * weight_m_corr
            / (np.sqrt(weight_c_corr) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * bias_m_corr
            / (np.sqrt(bias_c_corr) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


# ====================================================================
# Baseline Model (Stage 1)
# ====================================================================


class Model:
    """
    Model orchestrator for the baseline backpropagation network.

    Usage:
        model = Model()
        model.add(Layer_Dense(784, 128))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(128, 10))
        model.add(Activation_Softmax())
        model.set(loss=..., optimizer=...)
        model.finalize()
        history = model.train(X, y, ...)
    """

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(
            self.loss, Loss_CategoricalCrossentropy
        ):
            self.softmax_classifier_output = (
                Activation_Softmax_Loss_CategoricalCrossentropy()
            )

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(
        self,
        X_train,
        y_train,
        *,
        epochs=50,
        batch_size=128,
        validation_data=None,
        print_every=5,
    ):
        """Full training loop for the baseline MLP."""
        history = {"train_loss": [], "train_acc": [], "test_acc": []}
        n_samples = len(X_train)
        n_steps = (n_samples + batch_size - 1) // batch_size

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n_samples)
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]

            epoch_loss = 0.0
            epoch_correct = 0

            for step in range(n_steps):
                lo = step * batch_size
                hi = min(lo + batch_size, n_samples)
                X_b = X_shuf[lo:hi]
                y_b = y_shuf[lo:hi]

                output = self.forward(X_b, training=True)
                loss = self.loss.calculate(output, y_b)
                preds = np.argmax(output, axis=1)

                epoch_loss += loss
                epoch_correct += np.sum(preds == y_b)

                self.backward(output, y_b)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

            train_loss = epoch_loss / n_steps
            train_acc = epoch_correct / n_samples
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if validation_data is not None:
                X_v, y_v = validation_data
                v_out = self.forward(X_v, training=False)
                test_acc = float(np.mean(np.argmax(v_out, axis=1) == y_v))
                history["test_acc"].append(test_acc)

            if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
                msg = (
                    f"  Epoch {epoch:>3}/{epochs} | "
                    f"Loss {train_loss:.4f} | "
                    f"Train Acc {train_acc:.4f}"
                )
                if validation_data is not None:
                    msg += f" | Test Acc {test_acc:.4f}"
                print(msg)

        return history


# ====================================================================
# Neuromodulated Hebbian Network (Stage 2)
# ====================================================================


def train_hebbian(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    epochs=50,
    batch_size=128,
    learning_rate=0.01,
    use_ach=True,
    use_feedback_alignment=True,
    weight_clip=3.0,
    print_every=5,
):
    """
    Train a neuromodulated Hebbian network.

    Architecture: 784 -> 128 (ReLU) -> 10 (Softmax)

    Neuromodulator signals
    ----------------------
    Dopamine (DA) — modelled after midbrain RPE signals:
      * Output layer:  The error vector (target - prediction) acts as a
        per-neuron DA signal, encoding "the difference between the
        network's prediction and the ground-truth one-hot vector" (paper).
      * Hidden layer:  The output error is projected through fixed random
        feedback weights to give each hidden neuron a teaching signal.
        This avoids the weight-transport problem (Lillicrap et al., 2020).

    Acetylcholine (ACh) — modelled after basal-forebrain cholinergic gating:
      Scales with how poorly the model is performing (1 - accuracy).
      High early in training (permits plasticity), low once the network
      is accurate (gates learning, prevents runaway potentiation).

    Parameters
    ----------
    use_ach : bool
        If False, ACh is fixed at 1.0 (ablation — shows weight instability).
    use_feedback_alignment : bool
        If True (default), use error-based output learning and feedback
        alignment for the hidden layer.  If False, use the pure scalar-DA
        Hebbian rule for both layers (paper's Eq. 1 literally), which
        demonstrates the weight instability without ACh.
    weight_clip : float or None
        Hard clip range for hidden-layer weights. None disables clipping.
    """
    np.random.seed(42)

    # Learning rates (output can be higher — delta rule is self-correcting)
    output_lr = learning_rate
    hidden_lr = learning_rate * 0.15

    hidden = Layer_Hebbian(784, 128, learning_rate=hidden_lr)
    output = Layer_Hebbian(128, 10, learning_rate=output_lr)
    relu = Activation_ReLU()
    softmax = Activation_Softmax()
    loss_fn = Loss_CategoricalCrossentropy()

    # He initialisation — gives meaningful hidden activations from the start
    hidden.weights = np.random.randn(784, 128) * np.sqrt(2.0 / 784)
    output.weights = np.random.randn(128, 10) * np.sqrt(2.0 / 128)

    # Fixed random feedback weights for the hidden-layer DA projection.
    # The output error is projected through B to give each hidden neuron
    # a local teaching signal (feedback alignment — Lillicrap et al.).
    if use_feedback_alignment:
        B_feedback = np.random.randn(10, 128) * np.sqrt(1.0 / 10)

    # Scalar DA baseline (used when feedback alignment is off)
    reward_baseline = 0.1  # random chance

    # One-hot lookup
    eye10 = np.eye(10)

    # Running accuracy for ACh computation
    running_acc = 0.1  # start at random-chance

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "dopamine": [],
        "ach": [],
        "weight_snapshots": [],
        "max_weight": [],
    }

    n_samples = len(X_train)
    n_steps = (n_samples + batch_size - 1) // batch_size

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n_samples)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_da = []
        epoch_ach = []

        for step in range(n_steps):
            lo = step * batch_size
            hi = min(lo + batch_size, n_samples)
            X_b = X_shuf[lo:hi]
            y_b = y_shuf[lo:hi]
            bs = hi - lo

            # ---- Forward pass ----
            hidden.forward(X_b)
            relu.forward(hidden.output)
            output.forward(relu.output)
            softmax.forward(output.output)

            probs = softmax.output  # (bs, 10)

            # ---- Monitoring ----
            loss = loss_fn.calculate(probs, y_b)
            pred_classes = np.argmax(probs, axis=1)
            batch_acc = float(np.mean(pred_classes == y_b))
            epoch_loss += loss
            epoch_correct += int(np.sum(pred_classes == y_b))

            # ============================================================
            # DOPAMINE SIGNAL  (Reward Prediction Error)
            # ============================================================
            y_onehot = eye10[y_b]  # (bs, 10)
            output_error = y_onehot - probs  # (bs, 10)

            # ============================================================
            # ACETYLCHOLINE SIGNAL  (Plasticity Gate)
            # ============================================================
            running_acc = 0.995 * running_acc + 0.005 * batch_acc
            if use_ach:
                ach = float(0.3 + 0.7 * np.clip(1.0 - running_acc, 0, 1))
            else:
                ach = 1.0

            epoch_da.append(float(np.mean(np.abs(output_error))))
            epoch_ach.append(ach)

            # ============================================================
            # HEBBIAN WEIGHT UPDATES  (no backpropagation)
            # ============================================================
            if use_feedback_alignment:
                # --- Error-based output + feedback-aligned hidden ---
                output.output = output_error
                output.update_weights(np.ones((bs, 1)), ach)

                hidden_da = np.dot(output_error, B_feedback)  # (bs, 128)
                hidden.output = relu.output
                hidden.update_weights(hidden_da, ach)
            else:
                # --- Pure scalar-DA Hebbian (paper Eq. 1 literally) ---
                correct_probs = probs[np.arange(bs), y_b]
                da_scalar = float(np.mean(correct_probs) - reward_baseline)
                reward_baseline = 0.99 * reward_baseline + 0.01 * np.mean(correct_probs)
                output.output = softmax.output
                hidden.output = relu.output
                da_arr = np.full((bs, 1), da_scalar)
                output.update_weights(da_arr, ach)
                hidden.update_weights(da_arr, ach)

            # ---- ACh-mediated weight homeostasis ----
            if use_ach:
                # Soft decay (stronger when network is confident)
                decay = 0.0001 * (1.0 - ach)
                hidden.weights *= 1.0 - decay
                output.weights *= 1.0 - decay
                # Hard clip on hidden layer only (output self-corrects)
                if weight_clip is not None:
                    hidden.weights = np.clip(hidden.weights, -weight_clip, weight_clip)

            # NaN guard (for no-ACh ablation)
            if np.any(np.isnan(hidden.weights)) or np.any(np.isnan(output.weights)):
                print(
                    f"  WARNING: weights diverged at epoch {epoch}, "
                    f"step {step}. Stopping early."
                )
                remaining = epochs - epoch + 1
                for k in (
                    "train_loss",
                    "train_acc",
                    "test_acc",
                    "dopamine",
                    "ach",
                    "max_weight",
                ):
                    history[k].extend([np.nan] * remaining)
                return history

        # ---- Epoch bookkeeping ----
        train_loss = epoch_loss / n_steps
        train_acc = epoch_correct / n_samples
        max_w = max(
            float(np.max(np.abs(hidden.weights))), float(np.max(np.abs(output.weights)))
        )

        # Test evaluation
        hidden.forward(X_test)
        relu.forward(hidden.output)
        output.forward(relu.output)
        softmax.forward(output.output)
        test_acc = float(np.mean(np.argmax(softmax.output, axis=1) == y_test))

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["dopamine"].append(np.mean(epoch_da))
        history["ach"].append(np.mean(epoch_ach))
        history["max_weight"].append(max_w)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history["weight_snapshots"].append(
                {
                    "epoch": epoch,
                    "hidden": hidden.weights.copy(),
                    "output": output.weights.copy(),
                }
            )

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  Epoch {epoch:>3}/{epochs} | "
                f"Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Test Acc {test_acc:.4f} | "
                f"DA {np.mean(epoch_da):.4f} | "
                f"ACh {np.mean(epoch_ach):.3f} | "
                f"MaxW {max_w:.2f}"
            )

    return history


# ====================================================================
# Plotting
# ====================================================================

FIG_DIR = "figures"


def _savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


def plot_learning_curves(baseline, hebbian, epochs):
    """Figure 1: Accuracy and loss comparison over training."""
    ep = np.arange(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Accuracy --
    ax1.plot(ep, baseline["test_acc"], "b-", lw=2, label="Baseline (Backprop)")
    ax1.plot(ep, hebbian["test_acc"], "r-", lw=2, label="Hebbian (ACh + DA)")
    ax1.axhline(0.1, color="grey", ls="--", lw=1, label="Random chance")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Classification Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -- Loss --
    ax2.plot(ep, baseline["train_loss"], "b-", lw=2, label="Baseline (Backprop)")
    ax2.plot(ep, hebbian["train_loss"], "r-", lw=2, label="Hebbian (ACh + DA)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Training Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Stage 1 vs Stage 2: Learning Dynamics", fontsize=14, y=1.02)
    _savefig(fig, "learning_curves.png")


def plot_weight_stability(hebb_ach, hebb_no_ach, epochs):
    """Figure 2: Weight distributions with and without ACh gating."""
    snaps_ach = hebb_ach["weight_snapshots"]
    snaps_no_ach = hebb_no_ach.get("weight_snapshots", [])

    n_cols = max(len(snaps_ach), len(snaps_no_ach))
    if n_cols == 0:
        return
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Row 0: With ACh
    for i, snap in enumerate(snaps_ach):
        ax = axes[0, i]
        w = snap["hidden"].flatten()
        ax.hist(w, bins=60, color="steelblue", alpha=0.8, edgecolor="none")
        ax.set_title(f"Epoch {snap['epoch']}", fontsize=10)
        ax.set_xlim(-2.5, 2.5)
        if i == 0:
            ax.set_ylabel("With ACh\n(count)")

    # Row 1: Without ACh
    for i, snap in enumerate(snaps_no_ach):
        ax = axes[1, i]
        w = snap["hidden"].flatten()
        finite = w[np.isfinite(w)]
        if len(finite) > 0:
            ax.hist(finite, bins=60, color="indianred", alpha=0.8, edgecolor="none")
            rng = max(abs(finite.min()), abs(finite.max()), 2.5)
            ax.set_xlim(-rng * 1.1, rng * 1.1)
        ax.set_title(f"Epoch {snap['epoch']}", fontsize=10)
        if i == 0:
            ax.set_ylabel("Without ACh\n(count)")

    # Hide unused subplots
    for row in range(2):
        data = snaps_ach if row == 0 else snaps_no_ach
        for j in range(len(data), n_cols):
            axes[row, j].axis("off")

    fig.suptitle("Weight Stability: Hidden-Layer Weight Distributions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "weight_stability.png")


def plot_max_weight(hebb_ach, hebb_no_ach, epochs):
    """Figure 3: Maximum absolute weight magnitude over training."""
    ep = np.arange(1, epochs + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ach_w = hebb_ach["max_weight"]
    no_ach_w = hebb_no_ach["max_weight"]

    ax.plot(ep[: len(ach_w)], ach_w, "b-", lw=2, label="Hebbian + ACh (bounded)")
    ax.plot(
        ep[: len(no_ach_w)],
        no_ach_w,
        "r-",
        lw=2,
        label="Hebbian without ACh (unbounded)",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max |weight|")
    ax.set_title("Weight Magnitude Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, "max_weight.png")


def plot_neuromodulators(hebbian, epochs):
    """Figure 4: Dopamine and Acetylcholine dynamics over training."""
    ep = np.arange(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ep, hebbian["dopamine"], "g-", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Dopamine (RPE)")
    ax1.set_title("Dopamine Signal Over Training")
    ax1.axhline(0, color="grey", ls="--", lw=1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, hebbian["ach"], "m-", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean ACh (Plasticity Gate)")
    ax2.set_title("Acetylcholine Signal Over Training")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Neuromodulator Dynamics", fontsize=14, y=1.02)
    _savefig(fig, "neuromodulator_dynamics.png")


def plot_accuracy_comparison(baseline_acc, hebbian_acc, no_ach_acc):
    """Figure 5: Bar chart comparing final test accuracies."""
    fig, ax = plt.subplots(figsize=(7, 5))
    models = ["Baseline\n(Backprop)", "Hebbian\n(ACh + DA)", "Hebbian\n(No ACh)"]
    accs = [baseline_acc, hebbian_acc, no_ach_acc]
    colors = ["steelblue", "seagreen", "indianred"]

    bars = ax.bar(
        models, accs, color=colors, width=0.5, edgecolor="black", linewidth=0.8
    )
    for bar, acc in zip(bars, accs):
        label = f"{acc:.1%}" if not np.isnan(acc) else "Diverged"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            label,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Test Accuracy")
    ax.set_title("Final Test Accuracy Comparison")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.1, color="grey", ls="--", lw=1, label="Random chance")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, "accuracy_comparison.png")


# ====================================================================
# Main
# ====================================================================


def main():
    EPOCHS = 50
    BATCH_SIZE = 128

    print("=" * 64)
    print("  Neuromodulated Hebbian Learning  ")
    print("  Reproducing: Nazriev, 2026       ")
    print("=" * 64)

    # ---- Data ----
    print("\n[1/5] Loading MNIST dataset ...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # ---- Stage 1: Baseline MLP ----
    print(f"\n[2/5] Stage 1 -- Baseline MLP (Backpropagation, {EPOCHS} epochs)")
    np.random.seed(42)
    model = Model()
    model.add(Layer_Dense(784, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-4),
    )
    model.finalize()
    t0 = time.time()
    baseline = model.train(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        print_every=5,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Stage 2: Hebbian + ACh + DA ----
    print(
        f"\n[3/5] Stage 2 -- Neuromodulated Hebbian Network (ACh + DA, {EPOCHS} epochs)"
    )
    t0 = time.time()
    hebbian = train_hebbian(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=0.01,
        use_ach=True,
        weight_clip=3.0,
        print_every=5,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Ablation: Hebbian without ACh ----
    print(f"\n[4/5] Ablation -- Hebbian WITHOUT ACh gating ({EPOCHS} epochs)")
    t0 = time.time()
    no_ach = train_hebbian(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=0.01,
        use_ach=False,
        use_feedback_alignment=True,
        weight_clip=None,
        print_every=5,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Plots ----
    print(f"\n[5/5] Generating figures ...")
    plot_learning_curves(baseline, hebbian, EPOCHS)
    plot_weight_stability(hebbian, no_ach, EPOCHS)
    plot_max_weight(hebbian, no_ach, EPOCHS)
    plot_neuromodulators(hebbian, EPOCHS)

    b_acc = baseline["test_acc"][-1]
    h_acc = hebbian["test_acc"][-1]
    na_acc = no_ach["test_acc"][-1] if no_ach["test_acc"] else np.nan
    plot_accuracy_comparison(b_acc, h_acc, na_acc)

    # ---- Summary ----
    print("\n" + "=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)
    print(f"  Baseline (Backprop)     Test Acc: {b_acc:.2%}")
    print(f"  Hebbian  (ACh + DA)     Test Acc: {h_acc:.2%}")
    na_str = f"{na_acc:.2%}" if not np.isnan(na_acc) else "Diverged"
    print(f"  Hebbian  (No ACh)       Test Acc: {na_str}")
    print()
    print("  Paper's reported results:")
    print("    Baseline : ~97.5%")
    print("    Hebbian  : ~89.2%")
    print("  Note: exact numbers depend on random seed & hyperparameters.")
    print()
    print(f"  Figures saved to ./{FIG_DIR}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
