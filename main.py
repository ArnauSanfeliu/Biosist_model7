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

def figure3(spec=PAPER.with_params(r=0.5, nu=0.7, delta=1.0),
            phi_range=np.arange(0.01, 10, 0.01),
            phi_orbits=np.linspace(5.68, 10, 14),
            phi_inset=np.arange(5.8, 25.01, 0.4)):
    p = spec.params
    phi_star = p['delta'] * (1 + p['nu'])                 # BP
    phi_hb = p['delta'] * (1 + p['nu']) / (1 - p['nu'])   # HB
    colors = {'1,0': '#1f3fbf', '0,0': 'black', '*,*': '#1e6b2f'}

    # ---------- equilibris agrupats per branca ----------
    branches = defaultdict(list)
    for name, (M, I), stable, phi in steady_state(spec, phi_range):
        if stable is None:
            continue
        if name == '*,*' and phi <= p['delta']:            # evita la singularitat a phi = delta
            continue
        branches[name].append((phi, M, I, stable))

    # ---------- cicles límit: simula i queda't amb la cua ----------
    def cycle(phi, t_final=1500, dt=0.05, tail=400):
        d = solve(spec.with_params(phi=phi), t_final=t_final, dt=dt)
        n = int(tail / dt)
        t, M, I = (np.array(d[k][-n:]) for k in ('t', 'M', 'I'))
        peaks = np.where((M[1:-1] > M[:-2]) & (M[1:-1] >= M[2:]))[0] + 1
        T = np.mean(np.diff(t[peaks]))
        return t, M, I, T

    # més dens a prop de la HB, on l'amplitud creix com sqrt(phi - phi_hb)
    phi_cycles = phi_hb + np.r_[np.geomspace(0.002, 0.3, 12), np.arange(0.4, 10 - phi_hb + 0.01, 0.2)]
    max_M, max_I = [], []
    for phi in phi_cycles:
        _, M, I, _ = cycle(phi, t_final=4000)
        max_M.append(M.max()); max_I.append(I.max())

    # ---------- figura ----------
    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(13, 10))
    axB.set_prop_cycle(color=['k', 'g', 'b', 'r', 'gold', 'c'])
    axD.set_prop_cycle(color=['k', 'g', 'b', 'r', 'gold', 'c'])

    for ax, idx in ((axA, 1), (axC, 2)):
        for name, rows in branches.items():
            rows = np.array(rows)
            phi, y, stable = rows[:, 0], rows[:, idx], rows[:, 3].astype(bool)
            uns = ~stable
            uns[1:] |= uns[:-1].copy(); uns[:-1] |= uns[1:].copy()   # tanca el forat als canvis d'estabilitat
            ax.plot(phi, np.where(stable, y, np.nan), '-', color=colors[name], lw=1.5)
            ax.plot(phi, np.where(uns, y, np.nan), '--', color=colors[name], lw=1.5)

        # màxims dels cicles (vermell), començant al punt HB
        M_hb = p['delta'] * p['nu'] / (phi_hb - p['delta'])
        I_hb = p['r'] * p['nu'] * (phi_hb - phi_star) / (phi_hb - p['delta'])**2
        y_hb = M_hb if idx == 1 else I_hb
        ax.plot(np.r_[phi_hb, phi_cycles], np.r_[y_hb, max_M if idx == 1 else max_I], 'r-', lw=1.5)

        # marcadors BP i HB
        y_bp = 1 if idx == 1 else 0
        ax.plot(phi_star, y_bp, 's', mfc='white', mec='k', ms=5)
        ax.plot(phi_hb, y_hb, 's', color='darkred', ms=5)
        ax.annotate('BP', (phi_star, y_bp), xytext=(-18, -14), textcoords='offset points')
        ax.annotate('HB', (phi_hb, y_hb), xytext=(-18, -14), textcoords='offset points')

        ax.set_xlim(0, 10); ax.set_xlabel('φ')
        ax.grid(ls=':', color='gray')

    axA.set_ylim(-0.05, 1.05); axA.set_ylabel('M')
    axC.set_ylim(-0.05, 0.52); axC.set_ylabel('I')

    # ---------- B, D: un període normalitzat per cada phi ----------
    for phi in phi_orbits:
        t, M, I, T = cycle(phi, t_final=1000, tail=200)
        i0 = np.argmax(M[:int(T / 0.05) + 1])            # primer màxim de M
        start = i0 - int(0.22 * T / 0.05)                 # col·loca'l a t/T = 0.22
        if start < 0:
            start += int(T / 0.05)
        sl = slice(start, start + int(T / 0.05) + 1)
        tt = (t[sl] - t[sl][0]) / T
        line, = axB.plot(tt, M[sl])
        axD.plot(tt, I[sl], color=line.get_color())

    for ax, lab in ((axB, 'M'), (axD, 'I')):
        ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
        ax.set_xlabel('t / T'); ax.set_ylabel(lab)
        ax.grid(ls=':', color='gray')

    # ---------- inset: freqüència 1/T vs phi ----------
    ins = axA.inset_axes([0.4, 0.5, 0.45, 0.4])
    ins.plot(phi_inset, [1 / cycle(ph, t_final=1000, tail=200)[3] for ph in phi_inset], 'r-')
    ins.set_xlabel('φ'); ins.set_ylabel('1/T')

    for ax, letter in zip((axA, axB, axC, axD), 'ABCD'):
        ax.text(0.97, 0.95, letter, transform=ax.transAxes, ha='right', va='top', fontsize=16)

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
