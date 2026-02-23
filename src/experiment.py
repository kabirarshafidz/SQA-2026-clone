# train and eval one model
from .models import get_model_by_name
from .metrics import compute_metrics

def run_one(config, data_bundles, base_model=None):
    # receive metadata
    X_train = data_bundles["X_train"]
    y_train = data_bundles["y_train"]
    X_test = data_bundles["X_test"]
    y_test = data_bundles["y_test"]

    # build the model
    if not base_model:
        model_name = config["model_name"]
        model = get_model_by_name(model_name)
    else:
        model = base_model

    # train the model
    model.fit(X_train, y_train)

    # eval predictions from metrics.py
    y_pred = model.predict(X_test)
    y_scores = []
    if not base_model and config["model_name"] == "svm":
        y_scores = model.decision_function(X_test)
    elif not base_model and model_name == "random_forest":
        y_scores = model.predict_proba(X_test)[:,1]
    else: # for vqc models
        y_scores = model.predict_proba(X_test)

    metrics = compute_metrics(y_test, y_pred, y_scores)

    # return metrics
    return model, metrics