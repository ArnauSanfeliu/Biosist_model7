from cModel import Model
from cBareBones import Op_get,Op_set_der,Op_prod,Op_prod,Op_sign,Op_sum,Op_div,Op_const 

class Equations:
    name: str
    model: Model
    step_list: list

    def __init__(self, model):
        self.name = 'paper_eq'
        self.model = model
        self.step_list = None
        self.step_list_assign()

    def step_list_assign(self):
        self.step_list = [
            #  M equation
            Op_get(model=self.model, name='r'),
            Op_get(model=self.model, name='M'),
            Op_prod(model=self.model),
            Op_const(model=self.model, value=1),
            Op_get(model=self.model, name='M'),
            Op_sign(model=self.model),
            Op_sum(model=self.model),
            Op_prod(model=self.model),

            Op_get(model=self.model, name='phi'), # ------ Numerator ---------
            Op_get(model=self.model, name='I'),
            Op_get(model=self.model, name='M'),
            Op_prod(model=self.model),
            Op_prod(model=self.model),

            Op_get(model=self.model, name='M'), # ------ Denominator --------
            Op_get(model=self.model, name='nu'),
            Op_sum(model=self.model),
            Op_div(model=self.model),

            Op_sign(model=self.model),
            Op_sum(model=self.model),
            Op_set_der(model=self.model, name='M'),

            #  I equation
            Op_get(model=self.model, name='phi'), # ------ Numerator ---------
            Op_get(model=self.model, name='I'),
            Op_get(model=self.model, name='M'),
            Op_prod(model=self.model),
            Op_prod(model=self.model),

            Op_get(model=self.model, name='M'), # ------ Denominator --------
            Op_get(model=self.model, name='nu'),
            Op_sum(model=self.model),
            Op_div(model=self.model),

            Op_get(model=self.model, name='delta'),
            Op_get(model=self.model, name='I'),
            Op_prod(model=self.model),

            Op_sign(model=self.model),
            Op_sum(model=self.model),

            Op_set_der(model=self.model, name='I')
        ]