import logging
import pandas as pd


def plot_importance(model, features):
    logging.info("Plotting feature importance...")

    # Retrieve feature importances
    feature_importances = model.feature_importance()
    feature_names = features.columns

    # Debugging statements
    logging.info(f"Feature importances length: {len(feature_importances)}")
    logging.info(f"Feature names length: {len(feature_names)}")
    logging.info(f"Feature importances: {feature_importances}")
    l = list(feature_names)
    l.sort()
    logging.info(f"Feature names: {l}")

    # Filter feature names to match feature importances
    if len(feature_importances) != len(feature_names):
        logging.warning(
            "Feature importances and feature names lengths do not match. Dropping unmatched features."
        )
        min_length = min(len(feature_importances), len(feature_names))
        feature_importances = feature_importances[:min_length]
        feature_names = feature_names[:min_length]

    # Create DataFrame for feature importances
    feature_imp = pd.DataFrame({"Value": feature_importances, "Feature": feature_names})

    feature_imp = feature_imp.sort_values(by="Value", ascending=False)
    return feature_imp


def features(data, loaded_model):
    logging.info("Extracting features...")

    cols = [col for col in data.columns if col not in ["date", "id", "sales", "year"]]
    logging.info(f"Initial columns: {cols}")

    y_train = data["sales"]
    X_train = data[cols]

    y_val = data["sales"]
    X_val = data[cols]

    feat_imp = plot_importance(loaded_model, X_train)
    importance_low = feat_imp[feat_imp["Value"] < 50]["Feature"].values

    logging.info(f"Low importance features: {importance_low}")

    imp_feats = [col for col in cols if col not in importance_low]
    logging.info(f"Important features: {imp_feats}")

    train = data.loc[~data.sales.isna()]
    y_train = train["sales"]
    X_train = train[imp_feats]

    test = data.loc[data.sales.isna()]
    X_test = test[imp_feats]

    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Test data shape: {X_test.shape}")

    return X_test
