from train_codec import run_name


def test_run_name_shows_hyperparameter_overrides_but_not_codec_or_tracking_choice():
    overrides = ["codec=pod", "codec.rank=250", "mlflow=server", "hydra.verbose=true"]

    assert run_name("pod", overrides) == "pod codec.rank=250"
