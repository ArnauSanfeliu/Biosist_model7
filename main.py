"""
Functions to graph different figure types: concentration over time,
concentration according to some parameter, steady state analysis,
bifurcation analysis, etc.
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from cEquations import PAPER #, TEST, CESC
from Solve import solve, steady_state

def figure3(spec=PAPER, phi_range=np.arange(0,10,0.05)):
    def split_branches(points):
        branches = defaultdict(list)
        for name, (M, I), stable, phi in points:
            if stable is not None:                    # skip points where the Jacobian was undefined
                branches[name].append((phi, M, I, stable))
        return {k: sorted(v) for k, v in branches.items()}   # sort by phi

    points = steady_state(spec, phi_range)
    branches = split_branches(points)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    for name, rows in branches.items():
        phi, M, I, stable = map(np.array, zip(*rows))
        stable = stable.astype(bool)
        for ax, y in zip(axes, (M, I)):
            line, = ax.plot(phi, np.where(stable, y, np.nan), '-', label=name)
            ax.plot(phi, np.where(~stable, y, np.nan), '--', color=line.get_color())

    for ax, lab in zip(axes, ('M*', 'I*')):
        ax.set_xlabel('φ'); ax.set_ylabel(lab); ax.legend()
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.1, 1.1)
        
    plt.tight_layout(); plt.show()



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
