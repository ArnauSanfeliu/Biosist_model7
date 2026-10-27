import matplotlib.pyplot as plt
import numpy as np
import copy

from cModel import Model, Vals, Param
from cEquations import Equations
from cBareBones import * 

def main():
    # Create the model class instance  
    # Define model parameters and state variables in dicts
    paper = Model()
    paper_eq = Equations(model=paper)
    paper.state_var_dict = {
        'M': Vals(value=30, derivative=0),
        'I': Vals(value=0, derivative=0),
    }

    paper.params_dict = {
        'r'    : Param(1),
        'phi'  : Param(1),
        'nu'   : Param(1),
        'delta': Param(1),
    }

    paper.variables = paper.state_var_dict | paper.params_dict | paper.probes_dict


    # Import equations as "stack operators" from cEquations
    paper.step_list = paper_eq.step_list.copy()
    # Define runtime parameters
    M = []
    dM = []

    I = []
    dI = []

    model = paper
    steps = 100
    dt = 0.05
    solver = 'RK4'

    # Run simulation
    for i in range(steps):
        model.execute_step(dt=dt, solver=solver)
        M.append(model.variables['M'].value)
        dM.append(model.variables['M'].derivative)
        I.append(model.variables['I'].value)
        dI.append(model.variables['I'].derivative)

    # Generate the plots 

    plt.subplot(2,2,1)
    plt.plot(np.arange(1,steps+1),M)
    plt.title('M en el temps')
    plt.xlabel('temps [t]')
    plt.ylabel('M')

    plt.subplot(2,2,2)
    plt.plot(M, dM)
    plt.title('dM respcte M')
    plt.xlabel('M')
    plt.ylabel('dM')

    plt.subplot(2,2,3)
    plt.plot(np.arange(1,steps+1),I)
    plt.title('I en el temps')
    plt.xlabel('temps [t]')
    plt.ylabel('I')

    plt.subplot(2,2,4)
    plt.plot(I, dI)
    plt.title('dI respcte I')
    plt.xlabel('I')
    plt.ylabel('dI')

    plt.tight_layout()
    plt.show()
if __name__=="__main__":
    main()