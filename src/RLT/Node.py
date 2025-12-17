from typing import Any, Dict, List, Optional


class Node:
    """A node in the Reinforcement Learning Tree.

    Parameters
    ----------
    features : Optional[List[int]]
        Indices of features used for the split at this node.
    threshold : Optional[float]
        Threshold value for the split.
    coefficients : Optional[List[float]]
        Coefficients for the linear combination of features.
    left : Optional["Node"]
        Left child node.
    right : Optional["Node"]
        Right child node.
    valeur : Optional[Any]
        Prediction value if this is a terminal node.
    probabilities : Optional[Dict[Any, float]]
        Class probabilities at this node (for classification tasks).
    """

    def __init__(
        self,
        features: Optional[List[int]] = None,
        threshold: Optional[float] = None,
        coefficients: Optional[List[float]] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        *,
        valeur: Optional[Any] = None,
        probabilities: Optional[Dict[Any, float]] = None,
    ):
        self.right = right
        self.left = left
        self.coefficients = coefficients
        self.valeur = valeur
        self.features = features
        self.threshold = threshold
        self.probabilities = probabilities

    def is_terminal(self) -> bool:
        """
        Check if the node is a terminal (leaf) node.

        Returns
        -------
        bool
            True if the node is terminal, False otherwise.
        """
        return self.valeur is not None
