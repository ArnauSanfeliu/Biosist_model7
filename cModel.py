"""The numerical model: variables, parameters and the stack of operations.

A circuit is compiled into `Model.step_list`, a flat list of operators
(cBareBones) that, executed in order on a stack, write the derivative of
every state variable. The solvers (Euler, RK4) only call that list.
"""

from __future__ import annotations
from copy import deepcopy
from typing import NamedTuple

class Variable():
    value: float
    def __init__(self, value: float):
        self.value = value

class Param(Variable):
    pass

class Vals(Variable):
    """State variable: current value, derivative and the value of the next step."""
    new_value: float
    derivative: float
    def __init__(self, value: float, derivative: float):
        super().__init__(value)
        self.new_value = None
        self.derivative = derivative

class BaseOperator():
    model: Model
    REPR_FIELDS = None

    def __init__(self, model):
        self.model = model

    def execute(self, stack):
        pass

    # --- Representation (for debugging: print(model.repr_steps())) -----
    def _repr_name(self):
        """Op_get -> Get, Op_prod -> Prod."""
        name = type(self).__name__
        if name.startswith("Op_"):
            name = name[3:]
        return name[:1].upper() + name[1:]

    def _repr_args(self):
        fields = self.REPR_FIELDS
        if fields is None:
            # `model` is the same for every operator: leave it out
            fields = [k for k in vars(self)
                      if k != "model" and not k.startswith("_")]
        return [repr(getattr(self, f)) for f in fields if hasattr(self, f)]

    def __repr__(self):
        return f"{self._repr_name()}({', '.join(self._repr_args())})"

class Model:
    state_var_dict: dict[Vals]
    variables: dict
    params_dict: dict[Param]
    probes_dict: dict[Vals]
    step_list: list[BaseOperator]

    def __init__(self):
        self.state_var_dict = {}
        self.params_dict = {}
        self.variables = {}
        self.step_list = []
        self.probes_dict = {}
        # ALGEBRAIC variables: recomputed from the state at every evaluation,
        # never integrated (free forms and complex of a rapid equilibrium).
        self.algebraic_dict = {}
        # Compiler internals (intermediate fluxes). Readable by the
        # operators through `variables`, but not plotted.
        self.aux_dict = {}
        self.time = Variable(0.0)
        self.step_count = 0

    def repr_steps(self): # Debugging
        """The stacked operations, one per line."""
        return "\n".join(f"{i+1:3}  {op!r}" for i, op in enumerate(self.step_list))

    def add_var(self, name, initial_val):
        self.state_var_dict[name] = Vals(initial_val, 0.0)

    def add_param(self, name, value):
        self.params_dict[name] = Param(value)

    def add_probe(self, name, value):
        self.probes_dict[name] = Vals(0,0)

    def add_algebraic(self, name, value=0.0):
        self.algebraic_dict[name] = Variable(value)

    def add_aux(self, name, value=0.0):
        self.aux_dict[name] = Variable(value)

    def click_vars(self):
        """Commit the values computed by the solver."""
        for var in self.state_var_dict.values():
            if var.new_value != None:
                var.value = var.new_value
                var.new_value = None

    def _t0(self, dt: float):
        return self.step_count * dt

    def execute_step(self, dt: float, solver: str):
        self.execute_solver(solver, dt)
        self.click_vars()
        self.step_count += 1
        self.time.value = self.step_count * dt
        self.solve_dS() # refresh probes and algebraic variables

    def execute_solver(self, solver: str, dt: float):
        if solver == 'Euler':
            self.Euler(dt)
        if solver == 'RK4':
            self.RK4(dt)

    def solve_dS(self):
        """Run the operator list once: writes every derivative."""
        stack = []
        for step in self.step_list:
            step.execute(stack)

    def Euler(self, dt: float):
        t0 = self._t0(dt)
        self.time.value = t0
        self.solve_dS()
        for key in self.state_var_dict.keys():
            self.state_var_dict[key].new_value = self.state_var_dict[key].value + dt*self.state_var_dict[key].derivative
        self.time.value = t0 + dt


    def copy_dict(self, dic: dict, which: str):
        if which == 'value':
            return {k: v.value for k, v in dic.items()}
        else:
            return {k: v.derivative for k, v in dic.items()}

    def RK4(self, dt: float):
        """Classic 4th-order Runge-Kutta: four derivative evaluations
        (t0, t0+dt/2, t0+dt/2, t0+dt) combined with weights 1-2-2-1."""
        t0 = self._t0(dt)
        self.time.value = t0
        work = self.copy_dict(self.state_var_dict, 'value') # s1
        d_list = []

        self.time.value = t0
        self.solve_dS() # dX1 set
        d_list.append(self.copy_dict(self.state_var_dict, 'derivative'))
        for key in self.state_var_dict.keys():
            self.state_var_dict[key].value = work[key] + dt/2*self.state_var_dict[key].derivative # s2

        self.time.value = t0 + dt/2
        self.solve_dS() # dX2 set
        d_list.append(self.copy_dict(self.state_var_dict, 'derivative'))
        for key in self.state_var_dict.keys():
            self.state_var_dict[key].value = work[key] + dt/2*self.state_var_dict[key].derivative # s3

        self.time.value = t0 + dt/2
        self.solve_dS() # dX3 set
        d_list.append(self.copy_dict(self.state_var_dict, 'derivative'))
        for key in self.state_var_dict.keys():
            self.state_var_dict[key].value = work[key] + dt*self.state_var_dict[key].derivative # s4

        self.time.value = t0 + dt
        self.solve_dS() # dX4 set
        d_list.append(self.copy_dict(self.state_var_dict, 'derivative'))


        for key in self.state_var_dict.keys():
            self.state_var_dict[key].new_value = work[key] + dt/6*(d_list[0][key] +
                                                                         2*d_list[1][key] +
                                                                         2*d_list[2][key] +
                                                                         d_list[3][key]
                                                                        )