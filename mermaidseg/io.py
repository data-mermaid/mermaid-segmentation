"""
title: mermaidseg.io
abstract: Module that contains input/output and config reading functionality.
author: Viktor Domazetoski
date: 21-09-2025

Classes:
    ConfigDict - A dictionary subclass that allows attribute-style access to dictionary keys.
"""


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

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigDict(value)
            self[key] = value

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)
