"""
title: mermaidseg.io
abstract: Module that contains input/output and config reading functionality.
author: Viktor Domazetoski
date: 20-10-2025

Classes:
    ConfigDict - A dictionary subclass that allows attribute-style access to dictionary keys.
Functions:
    load_config(config_path) - Load configuration from a YAML file.
    update_config(base_config, config) - Update a base configuration dictionary with values from another configuration dictionary.
    setup_config(config_path=None, config_base_path="configs/base.yaml") - Set up configuration by loading and merging base and custom config files.
    get_parser() - Create and configure an argument parser for semantic segmentation training.
    update_config_with_args(config, args) - Update configuration dictionary with command line arguments.
"""

import argparse
from typing import Any, Dict, Optional

import yaml


class ConfigDict(dict):
    """
    A dictionary subclass that allows attribute-style access to dictionary keys.
    This class recursively converts nested dictionaries into ConfigDict instances,
    enabling dot notation access to dictionary keys.
    Methods
    -------
    __init__(dictionary)
        Initializes the ConfigDict with the given dictionary, converting nested
        dictionaries to ConfigDict instances.
    __getattr__(attr)
        Allows access to dictionary keys as attributes.
    __setattr__(key, value)
        Allows setting dictionary keys as attributes.
    """

    def __init__(self, dictionary: Dict[str, Any]):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigDict(value)
            self[key] = value

    def __getattr__(self, attr: str) -> Any:
        return self.get(attr)

    def __setattr__(self, key: str, value: Any) -> None:
        self.__setitem__(key, value)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Parsed configuration dictionary from the YAML file.
    """

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def update_config(base_config: ConfigDict, config: ConfigDict) -> ConfigDict:
    """
    Update a base configuration dictionary with values from another configuration dictionary.
    This function performs a recursive update where nested dictionaries are merged rather
    than completely replaced. For non-dictionary values, the new value overwrites the
    existing one.
    Args:
        base_config (ConfigDict): The base configuration dictionary to be updated.
        config (ConfigDict): The configuration dictionary containing updates to apply.
    Returns:
        ConfigDict: A new configuration dictionary with the updated values. The original
                   base_config is not modified.
    """
    updated_config = base_config.copy()
    for key, value in config.items():
        if key in updated_config:
            if isinstance(value, dict):
                updated_config[key].update(value)
            else:
                updated_config[key] = value
        else:
            updated_config[key] = value
    return updated_config


def setup_config(
    config_path: Optional[str] = None, config_base_path: str = "configs/base.yaml"
):
    """
    Set up configuration by loading and merging base and custom config files.
    This function loads a base configuration file and optionally merges it with
    a custom configuration file. If no custom config path is provided, only the
    base configuration is returned.
    Args:
        config_path (str, optional): Path to the custom configuration file.
            If None, only the base configuration will be used. Defaults to None.
        config_base_path (str, optional): Path to the base configuration file.
            Defaults to "configs/base.yaml".
    Returns:
        ConfigDict: A configuration dictionary object containing the merged
            configuration settings. If config_path is None, contains only the
            base configuration.
    Note:
        The custom configuration will override any conflicting settings in the
        base configuration during the merge process.
    """

    base_config = load_config(config_base_path)
    if config_path is None:
        return ConfigDict(base_config)
    config = load_config(config_path)
    updated_config = update_config(base_config, config)
    return ConfigDict(updated_config)


def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure an argument parser for semantic segmentation training.
    Returns:
        argparse.ArgumentParser: Configured argument parser with the following options:
            - run-name (str): Name identifier for the training run
            - model (str): Name of the model to use for training
            - batch-size (int): Batch size for training
            - epochs (int): Number of training epochs
            - lr (float): Learning rate for optimization
            - model-checkpoint (str): Path to model checkpoint file
            - log-epochs (int): Frequency of logging (every N epochs)
            - config (str): Path to configuration file
    """
    parser = argparse.ArgumentParser(description="Semantic Segmentation Run")

    # model and dataset
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--model", type=str, help="model name")

    # training hyper params
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--epochs", type=int, help="number of epochs to train")
    parser.add_argument("--lr", type=float, help="learning rate")

    # checkpoint and log
    parser.add_argument("--model-checkpoint", type=str, help="path to model checkpoint")

    parser.add_argument("--log-epochs", type=int, help="log every log-epochs")

    parser.add_argument("--config", type=str, help="path to config file")
    return parser


def update_config_with_args(config: ConfigDict, args: argparse.Namespace) -> ConfigDict:
    """
    Update configuration dictionary with command line arguments.
    Updates the provided configuration dictionary with values from command line
    arguments if they are present. Only non-None argument values will override
    the corresponding configuration values.
    Args:
        config (ConfigDict): The configuration dictionary to update.
        args (argparse.Namespace): Command line arguments namespace containing
            potential override values.
    Returns:
        ConfigDict: The updated configuration dictionary with command line
            argument values applied.
    Note:
        The function modifies the following configuration paths if corresponding
        arguments are provided:
        - run_name: config["run_name"]
        - model: config["model"]["name"]
        - model_checkpoint: config["model"]["checkpoint"]
        - batch_size: config["data"]["batch_size"]
        - epochs: config["training"]["epochs"]
        - lr: config["training"]["optimizer"]["lr"]
        - log_epochs: config["logger"]["log_epochs"]
    """

    if args.run_name:
        config["run_name"] = args.run_name
    if args.model:
        config["model"]["name"] = args.model
    if args.model_checkpoint:
        if "model" not in config:
            config["model"] = ConfigDict({"name": "segformer"})
        config["model"]["checkpoint"] = args.model_checkpoint
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["optimizer"]["lr"] = args.lr
    if args.log_epochs:
        config["logger"]["log_epochs"] = args.log_epochs
    return config
