import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from trainModels.config import CatBoostTrainingConfig
from trainModels.splits import WalkForwardFold, get_target_horizon


optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class CatBoostTuningResult:
    model: "CalibratedThresholdClassifier"
    validation_predictions: pd.DataFrame
    trial_history: pd.DataFrame
    feature_importance: pd.DataFrame
    removed_features: dict[str, str]
    best_parameters: dict
    fold_boundaries: list[dict]


class CalibratedThresholdClassifier:
    """Pickle-friendly classifier with OOF calibration and a tuned cutoff."""

    def __init__(
        self,
        estimators: list[CatBoostClassifier],
        feature_names: list[str],
        categorical_features: list[str],
        positive_label: str,
        negative_label: str,
        calibrator: LogisticRegression | None,
        decision_threshold: float,
        metadata: dict | None = None,
    ):
        if not estimators:
            raise ValueError("At least one fitted estimator is required")
        self.estimators = list(estimators)
        self.estimator = self.estimators[0]
        self.feature_names_ = list(feature_names)
        self.categorical_features_ = list(categorical_features)
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.calibrator = calibrator
        self.decision_threshold = float(decision_threshold)
        self.metadata = metadata or {}
        self.classes_ = np.asarray(self.estimator.classes_)

    def _prepare_features(self, feature) -> pd.DataFrame | np.ndarray:
        if isinstance(feature, pd.DataFrame):
            missing = [name for name in self.feature_names_ if name not in feature]
            if missing:
                raise ValueError(f"Missing model features: {missing[:10]}")
            prepared = feature[self.feature_names_].copy()
            for column in self.categorical_features_:
                if column in prepared:
                    prepared[column] = prepared[column].fillna("Unknown").astype(str)
            return prepared
        array = np.asarray(feature)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"Expected {len(self.feature_names_)} features, received shape {array.shape}"
            )
        return array

    def predict_proba(self, feature) -> np.ndarray:
        prepared = self._prepare_features(feature)
        raw = np.mean(
            [estimator.predict_proba(prepared) for estimator in self.estimators],
            axis=0,
        )
        positive_index = int(np.where(self.classes_ == self.positive_label)[0][0])
        positive_probability = raw[:, positive_index]
        if self.calibrator is not None:
            clipped = np.clip(positive_probability, 1e-7, 1 - 1e-7)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            calibrator_positive_index = int(
                np.where(self.calibrator.classes_ == 1)[0][0]
            )
            positive_probability = self.calibrator.predict_proba(logits)[
                :, calibrator_positive_index
            ]

        result = np.empty_like(raw, dtype=float)
        result[:, positive_index] = positive_probability
        result[:, 1 - positive_index] = 1 - positive_probability
        return result

    def predict(self, feature) -> np.ndarray:
        probabilities = self.predict_proba(feature)
        positive_index = int(np.where(self.classes_ == self.positive_label)[0][0])
        return np.where(
            probabilities[:, positive_index] >= self.decision_threshold,
            self.positive_label,
            self.negative_label,
        )

    def get_params(self, deep: bool = True) -> dict:
        return {
            "decision_threshold": self.decision_threshold,
            "feature_names": list(self.feature_names_),
            "categorical_features": list(self.categorical_features_),
            "estimator_params": self.estimator.get_params(),
            "ensemble_size": len(self.estimators),
        }


def _prepare_feature_frame(
    data: pd.DataFrame,
    feature_names: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    result = data[feature_names].copy()
    for column in categorical_features:
        if column in result:
            result[column] = result[column].fillna("Unknown").astype(str)
    return result


def _event_uniqueness_weights(length: int, horizon: int) -> np.ndarray:
    if length == 0 or horizon <= 1:
        return np.ones(length, dtype=float)
    concurrency = np.convolve(
        np.ones(length, dtype=float),
        np.ones(horizon, dtype=float),
        mode="full",
    )
    weights = np.empty(length, dtype=float)
    for position in range(length):
        active = concurrency[position : position + horizon]
        weights[position] = float(np.mean(1.0 / np.maximum(active, 1.0)))
    return weights


def _training_weights(
    data: pd.DataFrame,
    target_column: str,
    recency_halflife: int,
) -> np.ndarray:
    horizon = get_target_horizon(target_column)
    weights = pd.Series(1.0, index=data.index, dtype=float)
    group_column = "Ticker" if "Ticker" in data else None
    groups = data.groupby(group_column, sort=False) if group_column else [(None, data)]
    for _, group in groups:
        ordered = group.sort_values("Date")
        uniqueness = _event_uniqueness_weights(len(ordered), horizon)
        weights.loc[ordered.index] *= uniqueness

    if recency_halflife > 0:
        ordered_dates = sorted(data["Date"].astype(str).unique())
        date_age = {date: len(ordered_dates) - 1 - i for i, date in enumerate(ordered_dates)}
        ages = data["Date"].astype(str).map(date_age).to_numpy(dtype=float)
        weights *= np.power(0.5, ages / float(recency_halflife))

    if "Ticker" in data:
        observations_per_date = data.groupby("Date")["Date"].transform("size")
        weights /= observations_per_date.to_numpy(dtype=float)

    values = weights.to_numpy(dtype=float)
    mean_weight = float(np.mean(values))
    return values / mean_weight if mean_weight > 0 else np.ones(len(data))


def _macro_ticker_auc(
    validation_data: pd.DataFrame,
    binary_target: np.ndarray,
    probability: np.ndarray,
) -> float:
    if "Ticker" not in validation_data:
        return float(roc_auc_score(binary_target, probability))
    scores = []
    tickers = validation_data["Ticker"].astype(str).to_numpy()
    for ticker in np.unique(tickers):
        mask = tickers == ticker
        if len(np.unique(binary_target[mask])) < 2:
            continue
        scores.append(roc_auc_score(binary_target[mask], probability[mask]))
    return float(np.median(scores)) if scores else float("nan")


def _validation_objective_score(
    validation_data: pd.DataFrame,
    target: pd.Series,
    probability: np.ndarray,
    positive_label: str,
) -> float:
    binary_target = (target.to_numpy() == positive_label).astype(int)
    pooled_auc = float(roc_auc_score(binary_target, probability))
    macro_auc = _macro_ticker_auc(validation_data, binary_target, probability)
    if not np.isfinite(macro_auc):
        macro_auc = pooled_auc
    average_precision = float(average_precision_score(binary_target, probability))
    return 0.65 * macro_auc + 0.20 * pooled_auc + 0.15 * average_precision


def _model_parameters(
    sampled_parameters: dict,
    positive_label: str,
    negative_label: str,
    config: CatBoostTrainingConfig,
    iterations: int | None = None,
    random_seed: int | None = None,
) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": iterations or config.max_iterations,
        "depth": int(sampled_parameters["depth"]),
        "learning_rate": float(sampled_parameters["learning_rate"]),
        "l2_leaf_reg": float(sampled_parameters["l2_leaf_reg"]),
        "random_strength": float(sampled_parameters["random_strength"]),
        "bagging_temperature": float(sampled_parameters["bagging_temperature"]),
        "rsm": float(sampled_parameters["rsm"]),
        "class_weights": {
            negative_label: 1.0,
            positive_label: float(sampled_parameters["positive_class_weight"]),
        },
        "random_seed": config.model_seed if random_seed is None else random_seed,
        "thread_count": config.thread_count,
        "allow_writing_files": False,
        "logging_level": "Silent",
    }


def _sample_parameters(trial: optuna.Trial) -> dict:
    return {
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 100.0, log=True),
        "random_strength": trial.suggest_float(
            "random_strength", 0.001, 10.0, log=True
        ),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "positive_class_weight": trial.suggest_float(
            "positive_class_weight", 1.0, 8.0, log=True
        ),
        "recency_halflife": trial.suggest_categorical(
            "recency_halflife", [0, 252, 504, 756]
        ),
    }


def _fit_validation_folds(
    data: pd.DataFrame,
    target_column: str,
    positive_label: str,
    negative_label: str,
    feature_names: list[str],
    categorical_features: list[str],
    folds: list[WalkForwardFold],
    sampled_parameters: dict,
    config: CatBoostTrainingConfig,
) -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    predictions = []
    best_iterations = []
    importance_rows = []
    cat_features = [name for name in categorical_features if name in feature_names]
    model_parameters = _model_parameters(
        sampled_parameters, positive_label, negative_label, config
    )

    for fold_number, fold in enumerate(folds):
        train_data = data.iloc[fold.train_indices]
        validation_data = data.iloc[fold.validation_indices]
        train_feature = _prepare_feature_frame(
            train_data, feature_names, cat_features
        )
        validation_feature = _prepare_feature_frame(
            validation_data, feature_names, cat_features
        )
        model = CatBoostClassifier(**model_parameters)
        model.fit(
            train_feature,
            train_data[target_column],
            sample_weight=_training_weights(
                train_data,
                target_column,
                int(sampled_parameters["recency_halflife"]),
            ),
            cat_features=cat_features,
            eval_set=(validation_feature, validation_data[target_column]),
            use_best_model=True,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose=False,
        )
        positive_index = int(np.where(model.classes_ == positive_label)[0][0])
        probability = model.predict_proba(validation_feature)[:, positive_index]
        fold_predictions = validation_data[["Date", target_column]].copy()
        if "Ticker" in validation_data:
            fold_predictions["Ticker"] = validation_data["Ticker"].values
        fold_predictions["Probability"] = probability
        fold_predictions["Fold"] = fold_number
        predictions.append(fold_predictions)
        best_iteration = model.get_best_iteration()
        best_iterations.append(
            int(best_iteration + 1 if best_iteration >= 0 else model.tree_count_)
        )
        importance_rows.append(
            pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": model.get_feature_importance(),
                    "Fold": fold_number,
                }
            )
        )

    return (
        pd.concat(predictions).sort_index(),
        best_iterations,
        pd.concat(importance_rows, ignore_index=True),
    )


def _stable_feature_selection(
    importance: pd.DataFrame,
    feature_names: list[str],
    categorical_features: list[str],
    max_features: int,
) -> tuple[list[str], pd.DataFrame]:
    summary = importance.groupby("Feature")["Importance"].agg(
        MeanImportance="mean",
        StdImportance="std",
        NonZeroRate=lambda values: float((values > 0).mean()),
    )
    summary["StdImportance"] = summary["StdImportance"].fillna(0.0)
    summary["StabilityScore"] = summary["MeanImportance"] / (
        summary["StdImportance"] + 1e-9
    )
    summary = summary.sort_values(
        ["NonZeroRate", "MeanImportance", "StabilityScore"], ascending=False
    ).reset_index()

    mandatory = [name for name in categorical_features if name in feature_names]
    if len(feature_names) <= max_features:
        selected = list(feature_names)
    else:
        eligible = summary.loc[summary["NonZeroRate"] >= 0.5, "Feature"].tolist()
        ranked = list(
            dict.fromkeys(eligible + summary["Feature"].tolist())
        )
        selected = list(dict.fromkeys(mandatory + ranked))[:max_features]
    summary["Selected"] = summary["Feature"].isin(selected)
    return selected, summary


def _fit_calibrator(
    target: pd.Series,
    probability: np.ndarray,
    positive_label: str,
) -> LogisticRegression | None:
    binary_target = (target.to_numpy() == positive_label).astype(int)
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=10120024)
    calibrator.fit(logits, binary_target)
    # Calibration must preserve the model's ranking. A negative Platt slope
    # usually means a noisy validation sample and would invert every forecast.
    return calibrator if float(calibrator.coef_[0, 0]) > 0 else None


def _apply_calibrator(
    calibrator: LogisticRegression | None,
    probability: np.ndarray,
) -> np.ndarray:
    if calibrator is None:
        return probability
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    positive_index = int(np.where(calibrator.classes_ == 1)[0][0])
    return calibrator.predict_proba(logits)[:, positive_index]


def _select_f1_threshold(binary_target: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.quantile(probability, np.linspace(0.05, 0.95, 181)))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        prediction = probability >= threshold
        true_positive = float(np.sum((binary_target == 1) & prediction))
        false_positive = float(np.sum((binary_target == 0) & prediction))
        false_negative = float(np.sum((binary_target == 1) & ~prediction))
        denominator = 2 * true_positive + false_positive + false_negative
        score = 0.0 if denominator == 0 else 2 * true_positive / denominator
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def optimize_catboost(
    development_data: pd.DataFrame,
    target_column: str,
    positive_label: str,
    negative_label: str,
    feature_names: list[str],
    categorical_features: list[str],
    folds: list[WalkForwardFold],
    config: CatBoostTrainingConfig,
    removed_features: dict[str, str] | None = None,
    study_storage_path: Path | None = None,
) -> CatBoostTuningResult:
    """Optimize on walk-forward validation, calibrate OOF, then fit once."""
    categorical_features = [
        name for name in categorical_features if name in feature_names
    ]

    def objective(trial: optuna.Trial) -> float:
        sampled = _sample_parameters(trial)
        model_parameters = _model_parameters(
            sampled, positive_label, negative_label, config
        )
        scores = []
        for fold_number, fold in enumerate(folds):
            train_data = development_data.iloc[fold.train_indices]
            validation_data = development_data.iloc[fold.validation_indices]
            train_feature = _prepare_feature_frame(
                train_data, feature_names, categorical_features
            )
            validation_feature = _prepare_feature_frame(
                validation_data, feature_names, categorical_features
            )
            model = CatBoostClassifier(**model_parameters)
            model.fit(
                train_feature,
                train_data[target_column],
                sample_weight=_training_weights(
                    train_data,
                    target_column,
                    int(sampled["recency_halflife"]),
                ),
                cat_features=categorical_features,
                eval_set=(validation_feature, validation_data[target_column]),
                use_best_model=True,
                early_stopping_rounds=config.early_stopping_rounds,
                verbose=False,
            )
            positive_index = int(np.where(model.classes_ == positive_label)[0][0])
            probability = model.predict_proba(validation_feature)[:, positive_index]
            scores.append(
                _validation_objective_score(
                    validation_data,
                    validation_data[target_column],
                    probability,
                    positive_label,
                )
            )
            trial.report(float(np.mean(scores)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores) - 0.10 * np.std(scores))

    sampler = optuna.samplers.TPESampler(seed=config.optimizer_seed)
    identity_columns = ["Date", target_column, *feature_names]
    identity_columns = list(dict.fromkeys(identity_columns))
    study_identity = hashlib.sha256()
    study_identity.update(
        pd.util.hash_pandas_object(
            development_data[identity_columns], index=False
        ).values.tobytes()
    )
    study_config = config.to_dict()
    for non_objective_setting in (
        "n_trials",
        "tune_timeout_seconds",
        "thread_count",
        "ensemble_size",
        "max_features",
    ):
        study_config.pop(non_objective_setting, None)
    study_identity.update(
        json.dumps(study_config, sort_keys=True, default=str).encode("utf-8")
    )
    study_identity.update("|".join(feature_names).encode("utf-8"))
    study_name = f"catboost-{study_identity.hexdigest()[:20]}"
    storage = None
    if study_storage_path is not None:
        study_storage_path = Path(study_storage_path).resolve()
        study_storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{study_storage_path}"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=max(10, config.n_trials // 5),
            n_warmup_steps=1,
        ),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    remaining_trials = max(0, config.n_trials - len(study.trials))
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=config.tune_timeout_seconds,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("Hyperparameter optimization did not complete a trial")
    best_parameters = dict(study.best_trial.params)

    initial_oof, initial_best_iterations, initial_importance = _fit_validation_folds(
        development_data,
        target_column,
        positive_label,
        negative_label,
        feature_names,
        categorical_features,
        folds,
        best_parameters,
        config,
    )
    selected_features, importance_summary = _stable_feature_selection(
        initial_importance,
        feature_names,
        categorical_features,
        config.max_features,
    )
    selected_categorical = [
        name for name in categorical_features if name in selected_features
    ]

    if selected_features == feature_names:
        oof_predictions = initial_oof
        best_iterations = initial_best_iterations
    else:
        oof_predictions, best_iterations, _ = _fit_validation_folds(
            development_data,
            target_column,
            positive_label,
            negative_label,
            selected_features,
            selected_categorical,
            folds,
            best_parameters,
            config,
        )

    calibrator = _fit_calibrator(
        oof_predictions[target_column],
        oof_predictions["Probability"].to_numpy(),
        positive_label,
    )
    calibrated_probability = _apply_calibrator(
        calibrator, oof_predictions["Probability"].to_numpy()
    )
    oof_predictions["Raw Probability"] = oof_predictions["Probability"]
    oof_predictions["Probability"] = calibrated_probability
    binary_oof_target = (
        oof_predictions[target_column].to_numpy() == positive_label
    ).astype(int)
    decision_threshold = _select_f1_threshold(
        binary_oof_target, calibrated_probability
    )

    final_iterations = int(
        min(
            config.max_iterations,
            max(50, round(float(np.median(best_iterations)) * 1.10)),
        )
    )
    final_feature = _prepare_feature_frame(
        development_data, selected_features, selected_categorical
    )
    final_weights = _training_weights(
        development_data,
        target_column,
        int(best_parameters["recency_halflife"]),
    )
    final_models = []
    for ensemble_offset in range(config.ensemble_size):
        final_parameters = _model_parameters(
            best_parameters,
            positive_label,
            negative_label,
            config,
            iterations=final_iterations,
            random_seed=config.model_seed + ensemble_offset,
        )
        final_model = CatBoostClassifier(**final_parameters)
        final_model.fit(
            final_feature,
            development_data[target_column],
            sample_weight=final_weights,
            cat_features=selected_categorical,
            verbose=False,
        )
        final_models.append(final_model)
    wrapper = CalibratedThresholdClassifier(
        estimators=final_models,
        feature_names=selected_features,
        categorical_features=selected_categorical,
        positive_label=positive_label,
        negative_label=negative_label,
        calibrator=calibrator,
        decision_threshold=decision_threshold,
        metadata={
            "best_parameters": best_parameters,
            "final_iterations": final_iterations,
            "validation_objective": float(study.best_value),
            "study_name": study_name,
            "study_storage_path": (
                str(study_storage_path) if study_storage_path is not None else None
            ),
            "config": config.to_dict(),
            "ensemble_seeds": [
                config.model_seed + offset for offset in range(config.ensemble_size)
            ],
        },
    )
    fold_boundaries = [
        {
            "train_end": fold.train_end,
            "validation_start": fold.validation_start,
            "validation_end": fold.validation_end,
        }
        for fold in folds
    ]
    return CatBoostTuningResult(
        model=wrapper,
        validation_predictions=oof_predictions,
        trial_history=study.trials_dataframe(),
        feature_importance=importance_summary,
        removed_features=removed_features or {},
        best_parameters={
            **best_parameters,
            "iterations": final_iterations,
            "decision_threshold": decision_threshold,
        },
        fold_boundaries=fold_boundaries,
    )
