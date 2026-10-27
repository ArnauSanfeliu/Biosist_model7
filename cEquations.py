from cModel import Model
from cBareBones import Op_get,Op_set_der,Op_prod,Op_prod,Op_sign,Op_sum,Op_div,Op_const 

def build_paper_step_list(m: Model) -> list:
    return [
        #  M equation
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

        #  I equation
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


def build_test_step_list(m: Model) -> list: #dI/dt = nu - delta*I
    return [
        Op_get(m, 'nu'),
        Op_get(m, 'I'),
        Op_get(m, 'delta'),
        Op_prod(m),
        Op_sign(m),
        Op_sum(m),
        Op_set_der(m,'I')

    ]