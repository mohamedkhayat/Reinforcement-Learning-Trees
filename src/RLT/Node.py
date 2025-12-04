class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        coefficient=None,
        left=None,
        right=None,
        *,
        valeur=None,
    ):
        self.right = right
        self.left = left
        self.coefficient = coefficient
        self.valeur = valeur
        self.feature = feature
        self.threshold = threshold

    def is_terminal(self):
        return self.valeur is not None