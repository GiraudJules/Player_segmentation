import yaml


def load_config(config_path='config.yaml'):
    """
    Load configuration from a specified YAML file.

    Parameters:
    - config_path: Path to the YAML configuration file.

    Returns:
    - Dictionary containing the configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
