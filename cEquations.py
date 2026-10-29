from cModelSpec import ModelSpec
from cBareBones import Op_get,Op_set_der,Op_prod,Op_sign,Op_sum,Op_div,Op_const 

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

PAPER = ModelSpec(
    name='paper',
    state_vars={'M': 1, 'I': 0.2},
    params={'r': 0.5, 'phi': 0.45, 'nu': 0.5, 'delta': 0.2},
    build_steps=build_paper_step_list,
)

def build_cesc_step_list(m) -> list:
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

        Op_set_der(m, name='I'),

        # D equation: dD/dt =  [a*(1-M)+b]*(1-D)
        Op_get(m, name='a'),
        Op_const(m, value=1),
        Op_get(m, name='M'),
        Op_sign(m),
        Op_sum(m),
        Op_prod(m),
        Op_get(m, name='b'),
        Op_sum(m),
        Op_const(m, value=1),
        Op_get(m, name='D'),
        Op_sign(m),
        Op_sum(m),
        Op_prod(m),
        Op_set_der(m, name='D')
    ]

CESC = ModelSpec(
    name='cesc',
    state_vars={'M': 1.0, 'I': 0.2, 'D': 0.0},
    params={'r': 0.5, 'phi': 0.45, 'nu': 0.5, 'delta': 0.2, 'a': 0.0, 'b': 0.0},
    build_steps=build_cesc_step_list,
)

def build_test_step_list(m) -> list:
    return [
        #  I equation:  dI/dt = nu - delta*I
        Op_get(m, 'nu'),
        Op_get(m, 'I'),
        Op_get(m, 'delta'),
        Op_prod(m),
        Op_sign(m),
        Op_sum(m),
        Op_set_der(m,'I')

    ]

TEST = ModelSpec(
    name='test',
    state_vars={'I': 0.2},
    params={'nu': 0.5, 'delta': 0.2},
    build_steps=build_test_step_list,
)