# train and eval one model
from .models import get_model_by_name
from .metrics import compute_metrics

def run_one(config, data_bundles):
    # receive metadata
    model_name = config["model_name"]
    X_train = data_bundles["X_train"]
    y_train = data_bundles["y_train"]
    X_test = data_bundles["X_test"]
    y_test = data_bundles["y_test"]

    # build the model
    model = get_model_by_name(model_name)

    # train the model
    model.fit(X_train, y_train)

    # eval predictions from metrics.py
    y_pred = model.predict(X_test)
    y_scores = []
    if model_name == "svm":
        y_scores = model.decision_function(X_test)
    elif model_name == "random_forest":
        y_scores = model.predict_proba(X_test)[:,1]
    else: # for vqc models
        y_scores = model.predict_proba(X_test)


    metrics = compute_metrics(y_test, y_pred, y_scores)

    # return metrics
    return metrics