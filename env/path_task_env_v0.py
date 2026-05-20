"""
Contains the actual enviroment creation method with specified parameters.
"""

from .implementation import header as h
from .implementation.path_task_env import PathTaskMultiAgentEnv

def raw_env(args: h.EnvParams):
    """
    Method to generate the enviroment with specified
    enviroment arguments. 
    """
    return PathTaskMultiAgentEnv(args=args)
