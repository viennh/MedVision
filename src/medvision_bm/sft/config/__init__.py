from importlib.resources import files

import yaml


def load_model_info():
    config_path = files('medvision_bm.sft.config').joinpath('model_info.yaml')
    with config_path.open('r') as f:
        data = yaml.safe_load(f)
    return data['model_info']


def check_model_supported(model_name):
    model_info = load_model_info()
    if model_name not in model_info:
        raise ValueError(
            f"\n [Error] Model '{model_name}' is not supported. "
            f"Supported models are: {list(model_info.keys())}"
        )
