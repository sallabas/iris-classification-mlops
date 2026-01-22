import yaml
from pathlib import Path

def load_api_config():
    config_path = Path("irismodelproject/conf/base/api.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
