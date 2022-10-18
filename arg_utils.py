import argparse
import json


def create_parser_from_config(config):
    parser = argparse.ArgumentParser()
    for k, v in config.__dict__.items():
        if type(v) in [int, float, str]:
            parser.add_argument('--{}'.format(k), type=type(v))
        elif type(v) in [bool]:
            parser.add_argument('--{}'.format(k), type=bool)
    return parser


def override_config_from_arg(config, args):
    for k, v in args.__dict__.items():
        # print(v)
        if v is not None:
            setattr(config, k, v)
    return config


def dump_config(path, config):
    with open(path, 'w', newline='\n') as f:
        config_data_raw = config.__dict__
        config_data = {}
        for k, v in config_data_raw.items():
            if type(v) in [int, float, str, list, tuple, bool]:
                config_data[k] = v

        json.dump(config_data, f, indent=4)
