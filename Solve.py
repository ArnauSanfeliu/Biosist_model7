import matplotlib.pyplot as plt
import numpy as np

from cModel import Model, Vals, Param
from cEquations import build_paper_step_list, build_test_step_list


def solve(M0: float, I0: float, r: float, delta: float, nu: float, phi: float, model = build_paper_step_list) -> dict:
    # Create the model class instance
    # Define model parameters and state variables in dicts
    paper = Model()
    paper.state_var_dict = {
        'M': Vals(value=M0, derivative=0),
        'I': Vals(value=I0, derivative=0),
    }

    paper.params_dict = {
        'r'    : Param(r),
        'phi'  : Param(phi),
        'nu'   : Param(nu),
        'delta': Param(delta),
    }

    paper.variables = paper.state_var_dict | paper.params_dict | paper.probes_dict

    # Import equations as "stack operators" from cEquations
    paper.step_list = build_paper_step_list(paper)

    # Define runtime parameters
    model = paper
    t_final = 300.0    # total simulated time [months]
    dt = 0.05        # time step [months]
    solver = 'RK4'

    steps = int(round(t_final / dt))

    t = []
    M = []
    dM = []
    I = []
    dI = []

    # Run simulation
    for _ in range(steps):
        model.execute_step(dt=dt, solver=solver)
        t.append(model.time.value)
        M.append(model.variables['M'].value)
        dM.append(model.variables['M'].derivative)
        I.append(model.variables['I'].value)
        dI.append(model.variables['I'].derivative)

    result = {
        't': t,
        'M': M,
        'I': I,
        'dM': dM,
        'dI': dI,
    }

    return result