import matplotlib.pyplot as plt
import numpy as np

from cModel import Model, Vals, Param
from cEquations import build_paper_step_list, build_test_step_list


def main():
    # Create the model class instance
    # Define model parameters and state variables in dicts
    paper = Model()
    paper.state_var_dict = {
        'M': Vals(value=30, derivative=0),
        'I': Vals(value=0, derivative=0),
    }

    paper.params_dict = {
        'r'    : Param(1),
        'phi'  : Param(1),
        'nu'   : Param(1),
        'delta': Param(0.1),
    }

    paper.variables = paper.state_var_dict | paper.params_dict | paper.probes_dict

    # Import equations as "stack operators" from cEquations
    paper.step_list = build_test_step_list(paper)

    # Define runtime parameters
    model = paper
    t_final = 50.0    # total simulated time [s]
    dt = 0.05        # time step [s]
    solver = 'RK4'

    steps = int(round(t_final / dt))

    t = []
    M = []
    dM = []
    I = []
    dI = []

    # Run simulation
    for i in range(steps):
        model.execute_step(dt=dt, solver=solver)
        t.append(model.time.value)
        M.append(model.variables['M'].value)
        dM.append(model.variables['M'].derivative)
        I.append(model.variables['I'].value)
        dI.append(model.variables['I'].derivative)

    # Generate the plots
    plt.subplot(2, 2, 1)
    plt.plot(t, M)
    plt.title('M en el temps')
    plt.xlabel('temps [s]')
    plt.ylabel('M')

    plt.subplot(2, 2, 2)
    plt.plot(M, dM)
    plt.title('dM respecte M')
    plt.xlabel('M')
    plt.ylabel('dM')

    plt.subplot(2, 2, 3)
    plt.plot(t, I)
    plt.title('I en el temps')
    plt.xlabel('temps [s]')
    plt.ylabel('I')

    plt.subplot(2, 2, 4)
    plt.plot(I, dI)
    plt.title('dI respecte I')
    plt.xlabel('I')
    plt.ylabel('dI')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()