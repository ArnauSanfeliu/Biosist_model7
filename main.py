"""
Functions to graph different figure types: concentration over time,
concentration according to some parameter, steady state analysis,
bifurcation analysis, etc.
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from cEquations import PAPER, CESC
from Solve import solve, steady_state, limit_cycle, cycle_maxima
from cUtilities import load_patients

def figure3(spec=PAPER.with_params(r=0.5, eta=0.7, delta=1.0),
            param_range=np.arange(0.01, 10, 0.01),
            phi_orbits=np.linspace(5, 10, 14),
            phi_inset=np.arange(5.8, 25.01, 0.4)):
    p = spec.params
    phi_star = p['delta'] * (1 + p['eta'])                 # BP
    phi_hb = p['delta'] * (1 + p['eta']) / (1 - p['eta'])   # HB
    colors = {'1,0': '#1f3fbf', '0,0': 'black', '*,*': '#1e6b2f'}

    # ---------- equilibris agrupats per branca ----------
    branches = defaultdict(list)
    for name, (M, I), stable, phi in steady_state(spec, param_range, param_name='phi'):
        if stable is None:
            continue
        if name == '*,*' and phi <= p['delta']:            # evita la singularitat a phi = delta
            continue
        branches[name].append((phi, M, I, stable))

    # ---------- cicles límit ----------
    # més dens a prop de la HB, on l'amplitud creix com sqrt(phi - phi_hb)
    phi_cycles = phi_hb + np.r_[np.geomspace(0.002, 0.3, 12), np.arange(0.4, 10 - phi_hb + 0.01, 0.2)]
    maxima = cycle_maxima(spec, 'phi', phi_cycles)
    max_M, max_I = maxima['M'], maxima['I']

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
        M_hb = p['delta'] * p['eta'] / (phi_hb - p['delta'])
        I_hb = p['r'] * p['eta'] * (phi_hb - phi_star) / (phi_hb - p['delta'])**2
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
        out, T = limit_cycle(spec.with_params(phi=phi), t_final=1000, tail=200)
        t, M, I = out['t'], out['M'], out['I']
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
    ins.plot(phi_inset, [1 / limit_cycle(spec.with_params(phi=ph), t_final=1000, tail=200)[1] for ph in phi_inset], 'r-')
    ins.set_xlabel('φ'); ins.set_ylabel('1/T')

    for ax, letter in zip((axA, axB, axC, axD), 'ABCD'):
        ax.text(0.97, 0.95, letter, transform=ax.transAxes, ha='right', va='top', fontsize=16)

    plt.tight_layout(); plt.show()
        

def figure4(spec=PAPER.with_params(r=0.5, eta=0.7, delta=1.0, phi=3),
            param_range=np.arange(-0.1, 2.1, 0.01),
            eta_orbits=np.linspace(0.48, 0.08, 18)):
    p = spec.params
    eta_star = p['phi']/p['delta'] - 1                              # BP
    eta_hb = (p['phi'] - p['delta']) / (p['phi'] + p['delta'])      # HB
    colors = {'1,0': '#37409A', '0,0': '#231F20', '*,*': '#186634'}

    # ---------- equilibris agrupats per branca ----------
    branches = defaultdict(list)
    for name, (M, I), stable, eta in steady_state(spec, param_range, param_name='eta'):
        if stable is None:
            continue
        branches[name].append((eta, M, I, stable))

    # ---------- cicles límit: per a eta < eta_hb ----------
    # més dens a prop de la HB; no baixis de ~0.04 (a eta -> 0 la simulació divergeix)
    eta_cycles = eta_hb - np.r_[np.geomspace(0.002, 0.1, 10), np.arange(0.12, eta_hb - 0.04, 0.02)]
    max_M = cycle_maxima(spec, 'eta', eta_cycles, tail=600)['M']

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    axB.set_prop_cycle(color=['k', 'g', 'b', 'r', 'gold', 'c'])

    # ---------- A: diagrama de bifurcació ----------
    for name, rows in branches.items():
        rows = np.array(rows)
        eta, y, stable = rows[:, 0], rows[:, 1], rows[:, 3].astype(bool)
        uns = ~stable
        uns[1:] |= uns[:-1].copy(); uns[:-1] |= uns[1:].copy()   # tanca el forat als canvis d'estabilitat
        axA.plot(eta, np.where(stable, y, np.nan), '-', color=colors[name], lw=1.5)
        axA.plot(eta, np.where(uns, y, np.nan), '--', color=colors[name], lw=1.5)

    # màxims dels cicles (vermell), començant al punt HB
    M_hb = p['delta'] / (p['phi'] + p['delta'])
    axA.plot(np.r_[eta_hb, eta_cycles], np.r_[M_hb, max_M], 'r-', lw=1.5, color='#ED1E24')

    # marcadors BP i HB
    axA.plot(eta_star, 1, 's', mfc='white', mec='k', ms=5)
    axA.plot(eta_hb, M_hb, 's', color='darkred', ms=5)
    axA.annotate('BP', (eta_star, 1), xytext=(-20, 6), textcoords='offset points')
    axA.annotate('HB', (eta_hb, M_hb), xytext=(0, -14), textcoords='offset points')

    axA.set_xlim(-0.1, 2.1); axA.set_xlabel('η')
    axA.set_ylim(-0.05, 1.05); axA.set_ylabel('M')
    axA.grid(ls=':', color='gray')

    # ---------- B: PNA (període normalitzat i alineat) ----------
    dt = 0.05
    for eta in eta_orbits:
        out, T = limit_cycle(spec.with_params(eta=eta), t_final=2000, tail=600)
        t, M = out['t'], out['M']
        nT = int(T / dt)                                  # mostres per període
        i0 = np.argmax(M[:nT + 1])                        # 1) primer màxim de M
        start = i0 - int(0.22 * nT)                       # 2) col·loca'l a t/T = 0.22
        if start < 0:
            start += nT
        sl = slice(start, start + nT + 1)                 # 3) retalla un període
        tt = (t[sl] - t[sl][0]) / T                       # 4) normalitza el temps
        axB.plot(tt, M[sl])

    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.set_xlabel('t / T'); axB.set_ylabel('M')
    axB.grid(ls=':', color='gray')

    axA.text(0.97, 0.05, 'A', transform=axA.transAxes, ha='right', va='bottom', fontsize=16)
    axB.text(0.97, 0.95, 'B', transform=axB.transAxes, ha='right', va='top', fontsize=16)

    plt.tight_layout()
    plt.show()


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


def figure7(path='12918_2010_719_MOESM3_ESM.XLS',
            fits={3: dict(r=0.94, phi=0.50, delta=0.10, eta=0.67, I0=0.38),
                  6: dict(r=0.90, phi=0.55, delta=0.12, eta=0.64, I0=0.63),
                  8: dict(r=0.83, phi=0.38, delta=0.13, eta=0.48, I0=0.61),
                  9: dict(r=0.61, phi=0.43, delta=0.30, eta=0.14, I0=0.42)},
            noise=0.2, seed=0, spec=PAPER):
    data = load_patients(path)
    months = np.arange(0, 48)                 # el primer mes de dades es col·loca a t = 0, com al paper
    rng = np.random.default_rng(seed)
    colors = {'CEL': '#1f77b4', 'EDSS': '#8DC63F', 'RELAPSES': '#A020F0'}

    fig = plt.figure(figsize=(16, 10), layout='constrained')
    figA, figB = fig.subfigures(1, 2, width_ratios=[1.4, 1])

    # ---------- A: dades crues, 9 pacients x (CEL, EDSS, brots) ----------
    axesA = figA.subplots(len(data), 3)
    for i, p in enumerate(sorted(data)):
        for j, key in enumerate(('CEL', 'EDSS', 'RELAPSES')):
            ax = axesA[i, j]
            ax.plot(months, data[p][key], color=colors[key], lw=1)
            ax.set_title(f'Patient {p}', fontsize=8, pad=2)
            ax.set_ylabel(key.lower() if key == 'RELAPSES' else key, fontsize=7)
            ax.set_xlim(0, 48)
            if key == 'RELAPSES':
                ax.set_ylim(-0.05, 1.05)          # pacients sense brots: eix 0-1 igualment
            ax.tick_params(labelsize=6)
            if i == len(data) - 1:
                ax.set_xlabel('Months', fontsize=7)
    figA.text(0.0, 1.0, 'A', fontsize=16, va='top')

    # ---------- B: model ajustat vs CEL normalitzat ----------
    axesB = figB.subplots(len(fits), 1)
    for ax, (p, f) in zip(axesB, fits.items()):
        cel = data[p]['CEL'] / data[p]['CEL'].max()          # normalitza per pacient

        f = dict(f); I0 = f.pop('I0')
        # M0 = primer valor de CEL normalitzat (així comencen les corbes del paper); I0 ajustat
        d = solve(spec.with_params(**f).with_initial(M=cel[0], I=I0), t_final=48, dt=0.05)
        t, M = np.array(d['t']), np.array(d['M'])

        # soroll gaussià additiu als punts mensuals (sigma = 20% del màxim normalitzat)
        M_months = np.interp(months, t, M)
        M_noisy = M_months + rng.normal(0, noise * cel.max(), size=months.size)

        ax.plot(months, cel, 'o-', color='gray', mfc='white', ms=3, lw=0.8, label='CEL')
        ax.plot(months, M_noisy, '-', color='#1f77b4', lw=1.2, label='M (noise)')
        ax.plot(t, M, '--', color='#1f77b4', lw=1.5, label='M (no noise)')

        ax.set_title(f'Patient {p}', fontsize=9)
        ax.set_xlim(0, 48); ax.set_ylim(0, 1.05)
        ax.set_xlabel('Time (months)'); ax.set_ylabel('Myelin, CEL (normalised)')
        ax.text(1.03, 0.5, '\n'.join([f'r = {f["r"]:.2f}', f'φ = {f["phi"]:.2f}',
                                      f'δ = {f["delta"]:.2f}', f'η = {f["eta"]:.2f}',
                                      f'$I_0$ = {I0:.2f}']),
                transform=ax.transAxes, va='center', fontsize=9)
    axesB[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=8)
    figB.text(0.0, 1.0, 'B', fontsize=16, va='top')

    plt.tight_layout()
    plt.show()


def transient(spec=CESC, phis=(0.2, 0.45, 0.7), debug=False):
    plt.figure(figsize=(13, 5))
    for i, phi in enumerate(phis):
        if debug:
            data = solve(spec.with_params(phi=phi, a=0, b=0, Ta=10**12), t_final=600)
        else:
            data = solve(spec.with_params(phi=phi), t_final=600)


        t = data.get('t')
        M = data.get('M')
        I = data.get('I')
        D = data.get('D')
        G = np.array(M)*(1-np.array(D))

        plt.subplot(1, 3, i+1)
        plt.plot(t, M, label='[M]', color='#053E80')
        plt.plot(t, I, label='[I]', color='#75C8C0')
        plt.plot(t, D, label='[D]', color="#3E8A34")
        plt.plot(t, G, label='[G]', color="#B39124")
        plt.title('M,I,D,G fractions through time')
        plt.xlabel('Time (months)')
        plt.ylabel('[M],[I],[D],[G]')
        plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    figure2()
    figure3()
    # figure4()
    # figure5()
    # transient(debug=False)


if __name__ == '__main__':
    main()
