"""
Name             : Aman Banka
Registration No. : 261100660030
Course           : AME 5151 - Applied Linear Algebra Lab
Assignment       : Assignment 2 - Semantic Tagging with GloVe and NumPy
Institution      : Manipal School of Information Sciences, MAHE, Manipal
Faculty          : Dr. Devi Prasad
"""

"""
vector.py -- the tuple-based Vector class built earlier in the course.

Storage is an immutable tuple of Python floats. All arithmetic is done in
pure Python (no NumPy), so this class acts as an independent reference
implementation for checking the vectorised NumPy code.
"""

import math


class Vector:
    """An immutable vector in R^d stored as a tuple of floats."""

    def __init__(self, components):
        self._data = tuple(float(c) for c in components)
        if len(self._data) == 0:
            raise ValueError("A Vector must have at least one component.")

    # ----- basic container behaviour -------------------------------------
    @property
    def components(self):
        return self._data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"Vector({self._data})"

    def __eq__(self, other):
        return isinstance(other, Vector) and self._data == other._data

    def _check_dim(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector.")
        if len(self) != len(other):
            raise ValueError(
                f"Dimension mismatch: {len(self)} vs {len(other)}"
            )

    # ----- vector operations ---------------------------------------------
    def add(self, other):
        """u + v (component-wise)."""
        self._check_dim(other)
        return Vector(a + b for a, b in zip(self._data, other._data))

    def scale(self, c):
        """c * u for a scalar c."""
        return Vector(c * a for a in self._data)

    def dot(self, other):
        """u . v = sum_k u_k v_k."""
        self._check_dim(other)
        return sum(a * b for a, b in zip(self._data, other._data))

    def norm(self):
        """Euclidean norm ||u||_2 = sqrt(u . u)."""
        return math.sqrt(self.dot(self))

    def cosine_similarity(self, other):
        """cos(u, v) = (u . v) / (||u||_2 ||v||_2) for nonzero u, v."""
        self._check_dim(other)
        nu, nv = self.norm(), other.norm()
        if nu == 0.0 or nv == 0.0:
            raise ValueError("Cosine similarity is undefined for a zero vector.")
        return self.dot(other) / (nu * nv)

    # operator sugar
    __add__ = add

    def __mul__(self, c):
        return self.scale(c)

    __rmul__ = __mul__
