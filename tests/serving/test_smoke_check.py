from smoke_check import expected_from_uri


def test_a_version_uri_expects_its_name_and_version():
    assert expected_from_uri("models:/mhd-surrogate/3") == {
        "name": "mhd-surrogate",
        "version": "3",
    }


def test_an_alias_uri_expects_only_its_name():
    assert expected_from_uri("models:/mhd-surrogate@champion") == {"name": "mhd-surrogate"}


def test_a_logged_model_uri_expects_the_logged_model():
    assert expected_from_uri("models:/m-abc") == {"model_id": "m-abc"}
