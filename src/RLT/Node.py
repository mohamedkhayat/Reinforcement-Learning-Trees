class Node:
    """
    A node in the Reinforcement Learning Tree structure.

    Parameters
    ----------
    features : list of int, optional
        Indices of features used for the split at this node.
    threshold : float, optional
        Threshold value for the split.
    coefficients : list of float, optional
        Coefficients for the linear combination of features.
    left : Node, optional
        Left child node.
    right : Node, optional
        Right child node.
    valeur : Any, optional
        Prediction value if this is a terminal node.
    """

    def __init__(
        self,
        features=None,
        threshold=None,
        coefficients=None,
        left=None,
        right=None,
        *,
        valeur=None,
    ):
        self.right = right
        self.left = left
        self.coefficients = coefficients
        self.valeur = valeur
        self.features = features
        self.threshold = threshold

    def is_terminal(self) -> bool:
        """
        Check if the node is a terminal (leaf) node.

        Returns
        -------
        bool
            True if the node is terminal, False otherwise.
        """
        return self.valeur is not None
