import docseek


def test_package_exposes_version() -> None:
    assert docseek.__version__ == "0.1.0"
