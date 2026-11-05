"""
Functions to graph different figure types: concentration over time,
concentration according to some parameter, steady state analysis,
bifurcation analysis, etc.
"""

from cEquations import PAPER #, TEST, CESC

import matplotlib.pyplot as plt
import numpy as np

from Solve import solve, steady_state

def figure3(spec=PAPER, phi_range=np.arange(0,10,0.05)):
    steady_state(spec, phi_range)


def concentration_time():
    pass


def steady_state_analysis():
    pass


def figure2(spec=PAPER, phis=(0.2, 0.45, 0.7)):
    plt.figure(figsize=(13, 5))
    for i, phi in enumerate(phis):
        data = solve(spec.with_params(phi=phi))

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


def main():
    # figure2()
    figure3()


if __name__ == '__main__':
    main()
