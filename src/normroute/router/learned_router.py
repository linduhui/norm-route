"""Deterministic linear Router for leakage-safe Stage 5 experiments.

The model consumes only numeric Router feature bundles.  Evaluator labels and
expert scores are supplied to the training CLI as a separate supervision
channel and are never represented in this model's input schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


STAGE5_LINEAR_ROUTER_PROTOCOL_VERSION = "stage5.linear_router.v1"


class Stage5RouterError(ValueError):
    """Raised when a Stage 5 Router contract is invalid."""


@dataclass(frozen=True)
class LinearRouterModel:
    """Frozen standardized multinomial linear classifier."""

    feature_names: tuple[str, ...]
    experts: tuple[str, ...]
    location: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    l2: float
    training_seed: int
    backend: str
    protocol_version: str = STAGE5_LINEAR_ROUTER_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        dimension = len(self.feature_names)
        classes = len(self.experts)
        if dimension == 0 or classes < 2:
            raise Stage5RouterError(
                "linear Router requires non-empty features and at least two experts"
            )
        if len(set(self.feature_names)) != dimension:
            raise Stage5RouterError("linear Router feature names must be unique")
        if len(set(self.experts)) != classes:
            raise Stage5RouterError("linear Router expert names must be unique")
        if len(self.location) != dimension or len(self.scale) != dimension:
            raise Stage5RouterError("linear Router standardization shape mismatch")
        if len(self.weights) != dimension or any(
            len(row) != classes for row in self.weights
        ):
            raise Stage5RouterError("linear Router weight shape mismatch")
        if len(self.bias) != classes:
            raise Stage5RouterError("linear Router bias shape mismatch")
        numbers = (
            *self.location,
            *self.scale,
            *(value for row in self.weights for value in row),
            *self.bias,
            self.l2,
        )
        if not all(math.isfinite(float(value)) for value in numbers):
            raise Stage5RouterError("linear Router contains non-finite parameters")
        if any(float(value) <= 0.0 for value in self.scale):
            raise Stage5RouterError("linear Router scales must be positive")
        if self.l2 < 0.0:
            raise Stage5RouterError("linear Router l2 must be non-negative")

    def predict_proba(self, values: Any) -> Any:
        """Return stable expert probabilities for a numeric matrix."""

        np = _numpy()
        matrix = _matrix(values, "Router inference values", np)
        if matrix.shape[1] != len(self.feature_names):
            raise Stage5RouterError(
                "Router inference feature dimension disagrees with the model"
            )
        location = np.asarray(self.location, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        logits = ((matrix - location) / scale) @ weights + bias
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        denominator = np.sum(exp, axis=1, keepdims=True)
        if not bool(np.all(np.isfinite(denominator))) or bool(
            np.any(denominator <= 0.0)
        ):
            raise Stage5RouterError("Router probability normalization failed")
        probabilities = exp / denominator
        if not bool(np.all(np.isfinite(probabilities))):
            raise Stage5RouterError("Router produced non-finite probabilities")
        return probabilities

    def predict_indices(self, values: Any) -> Any:
        np = _numpy()
        return np.argmax(self.predict_proba(values), axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "feature_names": list(self.feature_names),
            "experts": list(self.experts),
            "location": list(self.location),
            "scale": list(self.scale),
            "weights": [list(row) for row in self.weights],
            "bias": list(self.bias),
            "l2": self.l2,
            "training_seed": self.training_seed,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LinearRouterModel":
        if value.get("protocol_version") != STAGE5_LINEAR_ROUTER_PROTOCOL_VERSION:
            raise Stage5RouterError("incompatible linear Router protocol")
        try:
            return cls(
                feature_names=tuple(str(item) for item in value["feature_names"]),
                experts=tuple(str(item) for item in value["experts"]),
                location=tuple(float(item) for item in value["location"]),
                scale=tuple(float(item) for item in value["scale"]),
                weights=tuple(
                    tuple(float(item) for item in row)
                    for row in value["weights"]
                ),
                bias=tuple(float(item) for item in value["bias"]),
                l2=float(value["l2"]),
                training_seed=int(value["training_seed"]),
                backend=str(value["backend"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Stage5RouterError("invalid linear Router artifact") from exc


@dataclass(frozen=True)
class LinearRouterFit:
    """Selected model and validation-only hyperparameter frontier."""

    model: LinearRouterModel
    frontier: tuple[dict[str, float], ...]


def fit_linear_router(
    train_values: Any,
    train_targets: Any,
    validation_values: Any,
    validation_targets: Any,
    *,
    feature_names: Sequence[str],
    experts: Sequence[str],
    l2_grid: Sequence[float] = (0.0, 1e-4, 1e-3),
    epochs: int = 40,
    batch_size: int = 1024,
    learning_rate: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> LinearRouterFit:
    """Fit candidates on train and select l2 using validation metrics only."""

    np = _numpy()
    train_x = _matrix(train_values, "training values", np)
    validation_x = _matrix(validation_values, "validation values", np)
    train_y = _targets(train_targets, train_x.shape[0], len(experts), np)
    validation_y = _targets(
        validation_targets, validation_x.shape[0], len(experts), np
    )
    names = tuple(str(item) for item in feature_names)
    expert_names = tuple(str(item) for item in experts)
    if train_x.shape[1] != len(names) or validation_x.shape[1] != len(names):
        raise Stage5RouterError("Router feature names disagree with matrix shape")
    if len(set(names)) != len(names) or len(set(expert_names)) != len(expert_names):
        raise Stage5RouterError("Router feature/expert names must be unique")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise Stage5RouterError("epochs must be a positive integer")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise Stage5RouterError("batch_size must be a positive integer")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
        raise Stage5RouterError("learning_rate must be positive and finite")
    candidates = _l2_grid(l2_grid)

    location = np.mean(train_x, axis=0, dtype=np.float64)
    scale = np.std(train_x, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    train_z = np.asarray((train_x - location) / scale, dtype=np.float32)
    validation_z = np.asarray(
        (validation_x - location) / scale, dtype=np.float32
    )
    class_weights = _balanced_class_weights(
        train_y, len(expert_names), np
    )

    fitted: list[tuple[float, float, float, Any, Any, str]] = []
    for l2 in candidates:
        # Candidate models receive the same initialization and minibatch order,
        # so validation differences isolate l2 instead of random shuffling.
        candidate_seed = int(seed)
        if device != "cpu":
            weights, bias, backend = _fit_torch(
                train_z,
                train_y,
                class_weights,
                classes=len(expert_names),
                l2=l2,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=float(learning_rate),
                seed=candidate_seed,
                device=device,
                np=np,
            )
        else:
            weights, bias = _fit_numpy(
                train_z,
                train_y,
                class_weights,
                classes=len(expert_names),
                l2=l2,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=float(learning_rate),
                seed=candidate_seed,
                np=np,
            )
            backend = "numpy"
        logits = validation_z @ weights + bias
        probabilities = _softmax(logits, np)
        predicted = np.argmax(probabilities, axis=1)
        accuracy = float(np.mean(predicted == validation_y))
        cross_entropy = float(
            -np.mean(
                np.log(
                    np.maximum(
                        probabilities[np.arange(validation_y.size), validation_y],
                        1e-12,
                    )
                )
            )
        )
        fitted.append(
            (accuracy, cross_entropy, l2, weights, bias, backend)
        )

    selected = min(
        fitted,
        key=lambda item: (-item[0], item[1], item[2]),
    )
    model = LinearRouterModel(
        feature_names=names,
        experts=expert_names,
        location=tuple(float(item) for item in location),
        scale=tuple(float(item) for item in scale),
        weights=tuple(
            tuple(float(item) for item in row) for row in selected[3]
        ),
        bias=tuple(float(item) for item in selected[4]),
        l2=float(selected[2]),
        training_seed=int(seed),
        backend=str(selected[5]),
    )
    frontier = tuple(
        {
            "l2": float(item[2]),
            "validation_accuracy": float(item[0]),
            "validation_cross_entropy": float(item[1]),
        }
        for item in sorted(fitted, key=lambda item: item[2])
    )
    return LinearRouterFit(model=model, frontier=frontier)


def _fit_numpy(
    values: Any,
    targets: Any,
    class_weights: Any,
    *,
    classes: int,
    l2: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    np: Any,
) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    weights = np.zeros((values.shape[1], classes), dtype=np.float32)
    bias = np.zeros(classes, dtype=np.float32)
    for _ in range(epochs):
        order = rng.permutation(values.shape[0])
        for start in range(0, values.shape[0], batch_size):
            indices = order[start : start + batch_size]
            batch_x = values[indices]
            batch_y = targets[indices]
            sample_weights = class_weights[batch_y]
            denominator = float(np.sum(sample_weights))
            logits = batch_x @ weights + bias
            probabilities = _softmax(logits, np)
            probabilities[np.arange(batch_y.size), batch_y] -= 1.0
            probabilities *= (sample_weights / denominator)[:, None]
            grad_weights = batch_x.T @ probabilities + l2 * weights
            grad_bias = np.sum(probabilities, axis=0)
            weights -= learning_rate * grad_weights
            bias -= learning_rate * grad_bias
    return (
        np.asarray(weights, dtype=np.float64),
        np.asarray(bias, dtype=np.float64),
    )


def _fit_torch(
    values: Any,
    targets: Any,
    class_weights: Any,
    *,
    classes: int,
    l2: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    np: Any,
) -> tuple[Any, Any, str]:
    try:
        import torch
    except ImportError as exc:
        raise Stage5RouterError(
            "torch is required when Router device is not cpu"
        ) from exc
    if not device.startswith("cuda:") or not torch.cuda.is_available():
        raise Stage5RouterError(
            "non-CPU Router training requires an available cuda:N device"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    target_device = torch.device(device)
    x = torch.as_tensor(values, dtype=torch.float32, device=target_device)
    y = torch.as_tensor(targets, dtype=torch.long, device=target_device)
    weights = torch.zeros(
        (values.shape[1], classes),
        dtype=torch.float32,
        device=target_device,
        requires_grad=True,
    )
    bias = torch.zeros(
        classes,
        dtype=torch.float32,
        device=target_device,
        requires_grad=True,
    )
    loss_weights = torch.as_tensor(
        class_weights, dtype=torch.float32, device=target_device
    )
    optimizer = torch.optim.Adam(
        (weights, bias), lr=learning_rate, weight_decay=l2
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(values.shape[0], generator=generator)
        for start in range(0, values.shape[0], batch_size):
            indices = order[start : start + batch_size].to(target_device)
            logits = x.index_select(0, indices) @ weights + bias
            loss = torch.nn.functional.cross_entropy(
                logits,
                y.index_select(0, indices),
                weight=loss_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return (
        weights.detach().cpu().numpy().astype(np.float64, copy=False),
        bias.detach().cpu().numpy().astype(np.float64, copy=False),
        f"torch:{device}",
    )


def _matrix(value: Any, name: str, np: Any) -> Any:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Stage5RouterError(f"{name} must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise Stage5RouterError(f"{name} must have shape [N,D]")
    if not bool(np.all(np.isfinite(matrix))):
        raise Stage5RouterError(f"{name} contains non-finite values")
    return matrix


def _targets(value: Any, rows: int, classes: int, np: Any) -> Any:
    targets = np.asarray(value)
    if targets.ndim != 1 or targets.shape[0] != rows:
        raise Stage5RouterError("Router targets must have shape [N]")
    if not bool(np.all(np.equal(targets, np.floor(targets)))):
        raise Stage5RouterError("Router targets must be integer class indices")
    targets = targets.astype(np.int64, copy=False)
    if bool(np.any(targets < 0)) or bool(np.any(targets >= classes)):
        raise Stage5RouterError("Router target class index is out of range")
    return targets


def _balanced_class_weights(targets: Any, classes: int, np: Any) -> Any:
    counts = np.bincount(targets, minlength=classes).astype(np.float64)
    weights = np.zeros(classes, dtype=np.float64)
    present = counts > 0
    weights[present] = targets.size / (float(classes) * counts[present])
    return weights


def _l2_grid(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise Stage5RouterError("l2_grid must be a sequence")
    try:
        converted = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise Stage5RouterError("l2_grid must contain finite values") from exc
    if (
        not converted
        or any(not math.isfinite(item) or item < 0.0 for item in converted)
        or tuple(sorted(set(converted))) != converted
    ):
        raise Stage5RouterError(
            "l2_grid must be non-empty, sorted, unique, finite, and non-negative"
        )
    return converted


def _softmax(logits: Any, np: Any) -> Any:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denominator = np.sum(exp, axis=1, keepdims=True)
    if not bool(np.all(np.isfinite(denominator))) or bool(
        np.any(denominator <= 0.0)
    ):
        raise Stage5RouterError("softmax normalization failed")
    return exp / denominator


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise Stage5RouterError("numpy is required for the Stage 5 Router") from exc
    return np


__all__ = [
    "LinearRouterFit",
    "LinearRouterModel",
    "STAGE5_LINEAR_ROUTER_PROTOCOL_VERSION",
    "Stage5RouterError",
    "fit_linear_router",
]
