from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CatBoostTrainingConfig:
    """Reproducible defaults for CatBoost model development."""

    n_trials: int = 80
    n_folds: int = 4
    validation_dates: int = 120
    min_train_dates: int = 252
    min_class_count: int = 5
    min_valid_folds: int = 2
    max_iterations: int = 3000
    early_stopping_rounds: int = 100
    optimizer_seed: int = 10120024
    model_seed: int = 10120024
    ensemble_size: int = 3
    thread_count: int = 1
    max_missing_fraction: float = 0.40
    max_features: int = 160
    top_fraction: float = 0.10
    tune_timeout_seconds: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)
