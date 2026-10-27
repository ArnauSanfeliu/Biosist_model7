# imports 

from cModel import Model, Vals, Param
# from cEquations import Equations


def main():
    # Create the model class instance  
    # Define model parameters and state variables in dicts
    paper = Model()
    paper.state_var_dict = {
        'M': Vals(value=0, derivative=0),
        'I': Vals(value=0, derivative=0),
    }

    paper.params_dict = {
        'r'    : Param(2),
        'phi'  : Param(2),
        'nu'   : Param(0),
        'delta': Param(0),
    }

    paper.variables = paper.state_var_dict | paper.params_dict | paper.probes_dict


    # Import equations as "stack operators" from cEquations
    paper.step_list()

    # Generate the plots 

if __name__=="__main__":
    main()