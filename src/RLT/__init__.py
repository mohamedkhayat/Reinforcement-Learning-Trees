"""
RLT - Reinforcement Learning Trees
A decision tree ensemble method with embedded variable importance.
"""

from RLT.Node import Node
from RLT.EmbeddedModel import EmbeddedModel
from RLT.ReinforcementLearningTree import ReinforcementLearningTree
from RLT.RLTRegression import RLTRegression
from RLT.RLTClassification import RLTClassification
from RLT.ReinforcementLearningTrees import ReinforcementLearningTrees

__version__ = "0.1.0"

__all__ = [
    "Node",
    "EmbeddedModel",
    "ReinforcementLearningTree",
    "RLTRegression",
    "RLTClassification",
    "ReinforcementLearningTrees",
]
