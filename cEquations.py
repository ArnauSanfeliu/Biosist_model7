from typing import Callable

from cModelSpec import ModelSpec
from cBareBones import Op_get,Op_set_der,Op_prod,Op_sign,Op_sum,Op_div,Op_const 
import numpy as np

def build_paper_step_list(m) -> list:
    return [
        #  M equation:  dM/dt = r*M*(1 - M) - phi*I*M/(M + nu)
        Op_get(m, name='r'),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_const(m, value=1),
        Op_get(m, name='M'),
        Op_sign(m),
        Op_sum(m),
        Op_prod(m),

        Op_get(m, name='phi'), # ------ Numerator ---------
        Op_get(m, name='I'),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_prod(m),

        Op_get(m, name='M'), # ------ Denominator --------
        Op_get(m, name='nu'),
        Op_sum(m),
        Op_div(m),

        Op_sign(m),
        Op_sum(m),
        Op_set_der(m, name='M'),

        #  I equation:  dI/dt = phi*I*M/(M + nu) - delta*I
        Op_get(m, name='phi'), # ------ Numerator ---------
        Op_get(m, name='I'),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_prod(m),

        Op_get(m, name='M'), # ------ Denominator --------
        Op_get(m, name='nu'),
        Op_sum(m),
        Op_div(m),

        Op_get(m, name='delta'),
        Op_get(m, name='I'),
        Op_prod(m),

        Op_sign(m),
        Op_sum(m),

        Op_set_der(m, name='I')
    ]

def build_jacobian_paper() -> Callable[[dict, dict], bool]:
    def jacobian(state: list, params: dict) -> bool | None:
        r, phi, nu, delta = params['r'], params['phi'], params['nu'], params['delta']
        M, I = state[0], state[1]

        with np.errstate(divide='ignore', invalid='ignore'):   # silence the warnings
            J = np.array([
                [r*(1 - 2*M) - phi*nu*I/(M + nu)**2,   -phi*M/(M + nu)],
                [phi*nu*I/(M + nu)**2,                  phi*M/(M + nu) - delta],
            ])

        if not np.all(np.isfinite(J)):
            return None                       # undefined at this point

        eig = np.linalg.eigvals(J)
        return bool(np.all(eig.real < 0))    
    return jacobian

def build_steady_states_paper() -> Callable[[dict],dict]:
    def steady_state(params:dict):
        delta, nu, phi, r = params.get('delta'), params.get('nu'), params.get('phi'), params.get('r')
        ss = {
            '0,0': [0,0], # M=0, I=0
            '1,0': [1,0], # M=1, I=0
            '*,*': [delta*nu/(phi-delta),r*nu*(phi-delta-delta*nu)/(phi-delta)**2]
        }
        return ss
    return steady_state


PAPER = ModelSpec(
    name='paper',
    state_vars={'M': 1, 'I': 0.2},
    params={'r': 0.5, 'phi': 0.45, 'nu': 0.5, 'delta': 0.2},
    build_steps=build_paper_step_list,
    jacobian=build_jacobian_paper(),
    steady_states=build_steady_states_paper(),
)

# def build_cesc_step_list(m) -> list:
#     return [
#         #  M equation:  dM/dt = r*M*(1 - M) - phi*I*M/(M + nu)
#         Op_get(m, name='r'),
#         Op_get(m, name='M'),
#         Op_prod(m),
#         Op_const(m, value=1),
#         Op_get(m, name='M'),
#         Op_sign(m),
#         Op_sum(m),
#         Op_prod(m),

#         Op_get(m, name='phi'), # ------ Numerator ---------
#         Op_get(m, name='I'),
#         Op_get(m, name='M'),
#         Op_prod(m),
#         Op_prod(m),

#         Op_get(m, name='M'), # ------ Denominator --------
#         Op_get(m, name='nu'),
#         Op_sum(m),
#         Op_div(m),

#         Op_sign(m),
#         Op_sum(m),
#         Op_set_der(m, name='M'),

#         #  I equation:  dI/dt = phi*I*M/(M + nu) - delta*I
#         Op_get(m, name='phi'), # ------ Numerator ---------
#         Op_get(m, name='I'),
#         Op_get(m, name='M'),
#         Op_prod(m),
#         Op_prod(m),

#         Op_get(m, name='M'), # ------ Denominator --------
#         Op_get(m, name='nu'),
#         Op_sum(m),
#         Op_div(m),

#         Op_get(m, name='delta'),
#         Op_get(m, name='I'),
#         Op_prod(m),

#         Op_sign(m),
#         Op_sum(m),

#         Op_set_der(m, name='I'),

#         # D equation: dD/dt =  [a*(1-M)+b]*(1-D)
#         Op_get(m, name='a'),
#         Op_const(m, value=1),
#         Op_get(m, name='M'),
#         Op_sign(m),
#         Op_sum(m),
#         Op_prod(m),
#         Op_get(m, name='b'),
#         Op_sum(m),
#         Op_const(m, value=1),
#         Op_get(m, name='D'),
#         Op_sign(m),
#         Op_sum(m),
#         Op_prod(m),
#         Op_set_der(m, name='D')
#     ]

# CESC = ModelSpec(
#     name='cesc',
#     state_vars={'M': 1.0, 'I': 0.2, 'D': 0.0},
#     params={'r': 0.5, 'phi': 0.45, 'nu': 0.5, 'delta': 0.2, 'a': 0.0, 'b': 0.0},
#     build_steps=build_cesc_step_list,
#     jacobian=build_jacobian_paper(),
# )

# def build_test_step_list(m) -> list:
#     return [
#         #  I equation:  dI/dt = nu - delta*I
#         Op_get(m, 'nu'),
#         Op_get(m, 'I'),
#         Op_get(m, 'delta'),
#         Op_prod(m),
#         Op_sign(m),
#         Op_sum(m),
#         Op_set_der(m,'I')

#     ]

# TEST = ModelSpec(
#     name='test',
#     state_vars={'I': 0.2},
#     params={'nu': 0.5, 'delta': 0.2},
#     build_steps=build_test_step_list,
#     jacobian=build_jacobian_paper(),
# )