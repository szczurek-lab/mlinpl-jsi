"""
Parameters for reward shaping functions.

Source: https://github.com/schwallergroup/saturn/blob/master/oracles/reward_shaping/function_parameters.py
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RewardShapingFunctionParameters:
    # Defaults to no transformation
    transformation_function: str = "no_transformation"
    parameters: Dict[str, float] = field(default_factory=dict)

