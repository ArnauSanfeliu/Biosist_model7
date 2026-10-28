import matplotlib.pyplot as plt

from Solve import solve


class Graphs:
    """
    Class so as to graph different figure types. There are different types of figures such as: concentration over time, 
    concentration according to some parameter, steady state analysis, bifurcation analysis, etc
    """
    @staticmethod 
    def bifurcation_diagram(): 
        #wasa
        pass

    @staticmethod 
    def concentration_time(): 
        pass
    
    @staticmethod 
    def steady_state_analysis(): 
        pass

    @staticmethod
    def figure2(M0: float = 1, I0: float = 0.2, r: float = 0.5, delta: float = 0.2, nu: float = 0.5, phi: list = [0.2,0.45,0.7], model = None):

        plt.figure(figsize=(13,5))
        for i in range(3):
            data = solve(M0,I0,r,delta,nu,phi[i],model)
            
            t = data.get('t')
            M = data.get('M')
            I = data.get('I')

            plt.subplot(1, 3, i+1)
            plt.plot(t, M, label='[M]', color='#053E80')
            plt.plot(t, I, label='[I]', color='#75C8C0')
            plt.title('M,I concentration through time')
            plt.xlabel('Time (months)')
            plt.ylabel('[M],[I]')
            plt.legend()
    
        plt.tight_layout()
        plt.show()
