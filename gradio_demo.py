from argparse import Namespace
from utils.demo_watermark import main
import yaml
import warnings
# Ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_config(file_path):
    """
    Load configuration settings from a YAML file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        Namespace: A namespace containing the configuration settings.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    # Path to the YAML configuration file
    config_path = 'config.yaml'

    # Load the YAML file to initialize the 'args' Namespace
    config = load_config(config_path)
    args = Namespace(**config)

    # Call the main function with the configuration
    main(args)
