"""
General tools for steady-state and stability analysis.

They do not depend on any particular model: each model only provides
its own definitions (see cEquations.py), and these functions turn them
into the callables that ModelSpec expects.
"""
from typing import Callable
import numpy as np


def _as_floats(values):
    """Convert a dict or a list to numpy floats, so that a division by
    zero gives inf/nan instead of raising ZeroDivisionError."""
    if isinstance(values, dict):
        return {k: np.float64(v) for k, v in values.items()}
    return [np.float64(v) for v in values]


def is_stable(J) -> bool | None:
    """Stability of an equilibrium from its Jacobian matrix.

    True  -> all eigenvalues have negative real part (stable)
    False -> at least one eigenvalue with real part >= 0 (unstable)
    None  -> the Jacobian is not defined at this point (inf/nan)
    """
    J = np.asarray(J, dtype=float)
    if not np.all(np.isfinite(J)):
        return None
    eig = np.linalg.eigvals(J)
    return bool(np.all(eig.real < 0))


def build_jacobian(matrix: Callable[[list, dict], list]) -> Callable[[list, dict], bool | None]:
    """From a model's Jacobian definition (state, params -> matrix)
    build the stability function (state, params -> bool | None)."""
    def jacobian(state: list, params: dict) -> bool | None:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            J = matrix(_as_floats(state), _as_floats(params))
        return is_stable(J)
    return jacobian


def build_steady_states(definition: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """From a model's steady-state definition (params -> {name: state})
    build the function that ModelSpec uses. Values are returned as floats;
    where an expression is undefined they are inf/nan."""
    def steady_states(params: dict) -> dict:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            ss = definition(_as_floats(params))
        return {name: [float(x) for x in state] for name, state in ss.items()}
    return steady_states
