import numpy as np

from cModelSpec import ModelSpec
from cUtilities import DotBar

def solve(spec: ModelSpec, t_final=300.0, dt=0.05, solver='RK4') -> dict:
        m = spec.build()
        result = {'t': []}
        result |= {k: [] for k in spec.state_vars}
        result |= {'d' + k: [] for k in spec.state_vars}
        result |= {k: [] for k in spec.probes}

        for _ in range(int(round(t_final / dt))):
            m.execute_step(dt=dt, solver=solver)
            result['t'].append(m.time.value)
            for k in spec.state_vars:
                result[k].append(m.variables[k].value)
                result['d' + k].append(m.variables[k].derivative)
            for k in spec.probes:
                result[k].append(m.variables[k].value)
        return result

def steady_state(spec: ModelSpec, param_range, param_name):
    
    if param_name not in spec.params.keys():
         raise KeyError
    
    result = []
    for param in param_range:
        states = spec.steady_states(params=spec.params | {param_name: param})
        for name,ss in states.items():
            stability = spec.jacobian(ss,spec.params | {param_name: param})
            result.append((name, ss, stability, param))
    return result


def limit_cycle(spec: ModelSpec, t_final: float = 1500, dt: float = 0.05,
                tail: float = 400, var: str = 'M') -> tuple[dict[str, np.ndarray], float]:
    """Simulate past the transient and return the last part of the trajectory.

    Args:
        spec: Model to simulate.
        t_final: Total simulation time (months).
        dt: Integration time step (months).
        tail: Length of the final window that is kept (months).
        var: State variable whose peaks are used to compute the period.

    Returns:
        A tuple ``(out, T)`` where:
            out: Final window of the time (``'t'``) and of each state variable.
            T: Mean period between consecutive peaks of ``var``, or ``nan`` if
                fewer than two peaks are found.
    """
    d = solve(spec, t_final=t_final, dt=dt)
    n = int(tail / dt)
    out = {k: np.array(d[k][-n:]) for k in ('t', *spec.state_vars)}
    x = out[var]
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    T = np.mean(np.diff(out['t'][peaks])) if len(peaks) > 1 else np.nan
    return out, T


def cycle_maxima(spec: ModelSpec, param_name, values, t_final=4000, dt=0.05, tail=400):
    """Màxim de cada variable al cicle límit per a cada valor del paràmetre
    (la corba vermella del diagrama de bifurcació)."""
    """.
    
        Args:
            spec: Model to simulate.
            param_name: parameter being modified for the analysis
            t_final: Total simulation time (months).
            dt: Integration time step (months).
            tail: Length of the final window that is kept (months).
    
        Returns:
            Dict: maximum array for each var 
        """
    maxima = {k: [] for k in spec.state_vars}
    for v in values:
        out, _ = limit_cycle(spec.with_params(**{param_name: v}), t_final, dt, tail)
        for k in spec.state_vars:
            maxima[k].append(out[k].max())
    return {k: np.array(m) for k, m in maxima.items()}

def period_grid(spec: ModelSpec, x_name: str, X: np.ndarray,
                y_name: str, Y: np.ndarray,
                t_final: float = 3000, dt: float = 0.05,
                tail: float = 300) -> np.ndarray:
    """Compute the limit-cycle period T at each point of a 2D parameter grid.

    The grid does not need to be rectangular: X and Y can come from any
    mapping of a meshgrid (e.g. a grid that follows the Hopf line).

    Args:
        spec: Model to simulate.
        x_name: Name of the parameter on the x axis.
        X: 2D array with the value of the x parameter at each grid point.
        y_name: Name of the parameter on the y axis.
        Y: 2D array with the value of the y parameter at each grid point
            (same shape as X).
        t_final: Total simulation time (months).
        dt: Integration time step (months).
        tail: Length of the final window used to measure the period (months).

    Returns:
        Array with the same shape as X with the period at each point, or
        ``nan`` where fewer than two peaks are found.
    """
    T = np.full(X.shape, np.nan)
    for i in DotBar(range(X.shape[0]), desc='rows', position=0):
        for j in DotBar(range(X.shape[1]), desc='  cols', position=1, leave=False):
            _, T[i, j] = limit_cycle(spec.with_params(**{x_name: X[i, j], y_name: Y[i, j]}),
                                     t_final=t_final, dt=dt, tail=tail)
    return T
