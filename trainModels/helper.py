import pickle
import pandas as pd
from pathlib import Path
from camel_converter import to_camel
from utils.pipeline import get_label_config

def _ensure_directories_exist(model_version, label_types):
    """
    (Internal Helper) Ensure all required directories exist before training.
    """
    for label_type in label_types:
        camel_label = to_camel(label_type)
        Path(f"data/stock/{model_version}/{camel_label}").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"data/stock/{model_version}/performance/{camel_label}").mkdir(
            parents=True, exist_ok=True
        )
    
    return

def _save_model(model_version, model, label_type, emiten, window):
    """
    (Internal Helper) Save a trained model to file.
    """
    camel_label = to_camel(label_type)
    filepath = f"data/stock/{model_version}/{camel_label}/{emiten}-{window}dd.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    
    return


def _combine_metrics(emiten, train_metrics, test_metrics, threshold):
    """
    (Internal Helper) Combine train and test metrics into a single DataFrame row.
    """
    train_df = pd.DataFrame(train_metrics)
    train_df.columns = [f"Train - {col}" for col in train_df.columns]

    test_df = pd.DataFrame(test_metrics)
    test_df.columns = [f"Test - {col}" for col in test_df.columns]

    result = pd.concat([train_df, test_df], axis=1)
    result.insert(0, "Kode", emiten)
    result["Threshold"] = threshold

    return result