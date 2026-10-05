from train import run_name


def test_run_name_shows_hyperparameter_overrides_but_not_model_or_tracking_choice():
    overrides = ["model=dmd", "model.rank=50", "mlflow=server", "mlflow.tracking_uri=x"]

    assert run_name("dmd", overrides) == "dmd model.rank=50"


def test_run_name_without_overrides_is_the_model_name():
    assert run_name("mean_field", []) == "mean_field"
