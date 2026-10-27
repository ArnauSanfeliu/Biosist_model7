from cModel import BaseOperator, Vals, Param, Model
import matplotlib.pyplot as plt

class Graphs:
    """
    Class so as to graph different figure types. There are different types of figures such as: concentration over time, 
    concentration according to some parameter, steady state analysis, bifurcation analysis, etc
    """
    model : Model

    


    def _init_(self):
        self.model = True

    def bifurcation_diagram(self): 
        #wasa
        pass

    def concentration_time(self): 
        pass

    def steady_state_analysis(self): 
        pass
