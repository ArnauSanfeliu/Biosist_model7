"""
Functions to graph different figure types: concentration over time,
concentration according to some parameter, steady state analysis,
bifurcation analysis, etc.
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Callable
from scipy.signal import find_peaks

from cEquations import PAPER, CESC, CESCH
from Solve import solve, steady_state, limit_cycle, cycle_maxima, period_grid
from cUtilities import load_patients, DotBar
from cModelSpec import ModelSpec

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


def _contour_labels(ax, cs):
    """Contour labels drawn on top of the line with a tight white box,
    instead of cutting the line (avoids big gaps on coarse grids)."""
    for t in ax.clabel(cs, inline=False, fmt=lambda v: f'T={v:g}', fontsize=8):
        t.set_bbox(dict(fc='white', ec='none', pad=0.3))


def figure6(levels=(28.3, 32, 35.7, 38.4, 41.2, 46, 47.3), n=22, r=0.5,
            delta_A=1.0, eta_B=0.1, t_final=1500, dt=0.2):
    """Iso-period curves of the limit cycles (paper, figure 6).

    The grid follows the Hopf line: for each eta (A) or delta (B), phi goes from
    just above phi_HB with log spacing, where the period changes fastest. The
    Hopf line itself is added with its analytical period T_H = 2*pi/sqrt(r*delta*eta).
    """
    phi_hb = lambda delta, eta: delta * (1 + eta) / (1 - eta)
    T_hb = lambda delta, eta: 2 * np.pi / np.sqrt(r * delta * eta)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # ---------- A: pla (phi, eta), delta fix ----------
    d = delta_A
    etas = np.linspace(0.02, 0.12, n)
    ETA, OFF = np.meshgrid(etas, np.geomspace(1e-3, 0.4, n), indexing='ij')
    PHI = phi_hb(d, ETA) + OFF                                  # sempre per sobre de Hopf
    TA = period_grid(PAPER.with_params(r=r, delta=d), 'phi', PHI, 'eta', ETA,
                     t_final=t_final, dt=dt)
    # afegeix la línia de Hopf (desplaçament 0) amb el període analític
    ETA, PHI, TA = np.c_[etas, ETA], np.c_[phi_hb(d, etas), PHI], np.c_[T_hb(d, etas), TA]
    _contour_labels(axA, axA.contour(PHI, ETA, TA, levels=levels, cmap='turbo'))
    eta_bp = (2*np.pi/np.array(levels))**2 / (r*d)              # BP: on T_H = T
    axA.plot(phi_hb(d, eta_bp), eta_bp, 's', mfc='white', mec='k')
    axA.set_xlim(1.05, 1.35); axA.set_ylim(0.02, 0.12)
    axA.set_xlabel('φ'); axA.set_ylabel('η'); axA.set_title(f'δ = {d}')

    # ---------- B: pla (delta, phi), eta fix ----------
    eta = eta_B
    deltas = np.linspace(0.2, 1.4, n)
    DEL, OFF = np.meshgrid(deltas, np.geomspace(1e-3, 1.0, n), indexing='ij')
    PHI = phi_hb(DEL, eta) + OFF
    TB = period_grid(PAPER.with_params(r=r, eta=eta), 'delta', DEL, 'phi', PHI,
                     t_final=t_final, dt=dt)
    DEL, PHI, TB = np.c_[deltas, DEL], np.c_[phi_hb(deltas, eta), PHI], np.c_[T_hb(deltas, eta), TB]
    _contour_labels(axB, axB.contour(DEL, PHI, TB, levels=levels, cmap='turbo'))
    d_bp = (2*np.pi/np.array(levels))**2 / (r*eta)
    axB.plot(d_bp, phi_hb(d_bp, eta), 's', mfc='white', mec='k')
    axB.set_xlim(0, 1.4); axB.set_ylim(0, 1.4)
    axB.set_xlabel('δ'); axB.set_ylabel('φ'); axB.set_title(f'η = {eta}')

    for ax in (axA, axB):
        ax.grid(ls=':', color='gray')
    plt.tight_layout(); plt.show()


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
    C_MODEL, C_CEL = '#1f77b4', '0.55'

    # figura més ampla + marges i separació entre columnes perquè el text de paràmetres respiri
    fig = plt.figure(figsize=(19, 11), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.25, h_pad=0.12, wspace=0.0, hspace=0.04)
    figA, figB = fig.subfigures(1, 2, width_ratios=[1.4, 1], wspace=0.08)

    def clean(ax):
        ax.set_xlim(0, 48)
        ax.set_xticks(range(0, 49, 12))
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)

    # ---------- A: dades crues, 9 pacients x (CEL, EDSS, brots) ----------
    axesA = figA.subplots(len(data), 3)
    for i, p in enumerate(sorted(data)):
        for j, key in enumerate(('CEL', 'EDSS', 'RELAPSES')):
            ax = axesA[i, j]
            ax.plot(months, data[p][key], color=colors[key], lw=1.1)
            ax.set_title(f'Patient {p}', fontsize=8, pad=2, loc='left', fontweight='bold')
            ax.set_ylabel(key.lower() if key == 'RELAPSES' else key, fontsize=7)
            if key == 'RELAPSES':
                ax.set_ylim(-0.05, 1.05)          # pacients sense brots: eix 0-1 igualment
            clean(ax)
            ax.tick_params(labelsize=6)
            if i == len(data) - 1:
                ax.set_xlabel('Months', fontsize=8)
    figA.suptitle('A', fontsize=15, fontweight='bold', x=0.02, ha='left')

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

        ax.plot(months, cel, 'o-', color=C_CEL, mfc='white', ms=3, lw=0.8, label='CEL')
        ax.plot(months, M_noisy, '-', color=C_MODEL, lw=1.1, alpha=0.75, label='M (noise)')
        ax.plot(t, M, '--', color=C_MODEL, lw=1.8, label='M (no noise)')

        ax.set_title(f'Patient {p}', fontsize=10, loc='left', fontweight='bold')
        ax.set_ylim(0, 1.05)
        clean(ax)
        ax.tick_params(labelsize=8)
        ax.text(1.04, 0.5, '\n'.join([f'r = {f["r"]:.2f}', f'φ = {f["phi"]:.2f}',
                                      f'δ = {f["delta"]:.2f}', f'η = {f["eta"]:.2f}',
                                      f'$I_0$ = {I0:.2f}']),
                transform=ax.transAxes, va='center', fontsize=8.5, linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.45', fc='0.97', ec='0.85', lw=0.6))
    axesB[-1].set_xlabel('Time (months)')
    axesB[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.38), ncol=3,
                     fontsize=8, frameon=False)
    figB.supylabel('Myelin, CEL (normalised)', fontsize=10)
    figB.suptitle('B', fontsize=15, fontweight='bold', x=0.02, ha='left')

    plt.show()


def comparaison_7(path='12918_2010_719_MOESM3_ESM.XLS',
            fits={3: dict(r=0.94, phi=0.50, delta=0.10, eta=0.67, I0=0.38),
                  6: dict(r=0.90, phi=0.55, delta=0.12, eta=0.64, I0=0.63),
                  8: dict(r=0.83, phi=0.38, delta=0.13, eta=0.48, I0=0.61),
                  9: dict(r=0.61, phi=0.43, delta=0.30, eta=0.14, I0=0.42)},
            noise=0.2, seed=0, paper=PAPER, cesc=CESC):
    data = load_patients(path)
    t_final = 48
    months = np.arange(0, t_final)                 # el primer mes de dades es col·loca a t = 0, com al paper
    rng = np.random.default_rng(seed)

    C_PAPER, C_EXT, C_CEL = '#1f77b4', '#ec8633', '0.55'

    # figura més ampla + marges i separació entre columnes perquè el text de paràmetres respiri
    fig = plt.figure(figsize=(19, 11), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.25, h_pad=0.12, wspace=0.0, hspace=0.04)
    figA, figB, figC = fig.subfigures(1, 3, width_ratios=[1, 1, 1.1], wspace=0.08)

    def style(ax, p, f, I0, t_final):
        ax.set_title(f'Patient {p}', fontsize=10, loc='left', fontweight='bold')
        ax.set_xlim(0, t_final); ax.set_ylim(0, 1.05)
        ax.set_xticks(range(0, 49, 12))
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.text(1.04, 0.5, '\n'.join([f'r = {f["r"]:.2f}', f'φ = {f["phi"]:.2f}',
                                      f'δ = {f["delta"]:.2f}', f'η = {f["eta"]:.2f}',
                                      f'$I_0$ = {I0:.2f}']),
                transform=ax.transAxes, va='center', fontsize=8.5, linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.45', fc='0.97', ec='0.85', lw=0.6))

    def finish(subfig, axes, title, ncol):
        axes[-1].set_xlabel('Time (months)')
        axes[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.38), ncol=ncol,
                        fontsize=8, frameon=False)
        subfig.supylabel('Myelin, CEL (normalised)', fontsize=10)
        subfig.suptitle(title, fontsize=15, fontweight='bold', x=0.02, ha='left')

    # ---------- A: dades crues, 9 pacients x (CEL, EDSS, brots) ----------
    axesA = figA.subplots(len(fits), 1)
    for ax, (p, f) in zip(axesA, fits.items()):
        cel = data[p]['CEL'] / data[p]['CEL'].max()          # normalitza per pacient

        f = dict(f); I0 = f.pop('I0')
        # M0 = primer valor de CEL normalitzat (així comencen les corbes del paper); I0 ajustat
        d = solve(paper.with_params(**f).with_initial(M=cel[0], I=I0), t_final=48, dt=0.05)
        t, M = np.array(d['t']), np.array(d['M'])

        # soroll gaussià additiu als punts mensuals (sigma = 20% del màxim normalitzat)
        M_months = np.interp(months, t, M)
        M_noisy = M_months + rng.normal(0, noise * cel.max(), size=months.size)

        ax.plot(months, cel, 'o-', color=C_CEL, mfc='white', ms=3, lw=0.8, label='CEL')
        ax.plot(months, M_noisy, '-', color=C_PAPER, lw=1.1, alpha=0.75, label='M (noise)')
        ax.plot(t, M, '--', color=C_PAPER, lw=1.8, label='M (no noise)')
        style(ax, p, f, I0, t_final)
    finish(figA, axesA, 'Paper', ncol=3)

    # ---------- B: model ajustat vs CEL normalitzat ----------
    axesB = figB.subplots(len(fits), 1)
    for ax, (p, f) in zip(axesB, fits.items()):
        cel = data[p]['CEL'] / data[p]['CEL'].max()          # normalitza per pacient

        f = dict(f); I0 = f.pop('I0')
        d = solve(cesc.with_params(**f).with_initial(M=cel[0], I=I0), t_final=48, dt=0.05)
        t, M = np.array(d['t']), np.array(d['M'])

        M_months = np.interp(months, t, M)
        M_noisy = M_months + rng.normal(0, noise * cel.max(), size=months.size)

        ax.plot(months, cel, 'o-', color=C_CEL, mfc='white', ms=3, lw=0.8, label='CEL')
        ax.plot(months, M_noisy, '-', color=C_EXT, lw=1.1, alpha=0.75, label='M (noise)')
        ax.plot(t, M, '--', color=C_EXT, lw=1.8, label='M (no noise)')
        style(ax, p, f, I0, t_final)
    finish(figB, axesB, 'Extension', ncol=3)

    # ---------- C: model comparaisons ----------
    t_final = 48
    months = np.arange(0, t_final)
    axesC = figC.subplots(len(fits), 1)
    for ax, (p, f) in zip(axesC, fits.items()):
        cel = data[p]['CEL'] / data[p]['CEL'].max()       

        f = dict(f); I0 = f.pop('I0')
        d_paper = solve(paper.with_params(**f).with_initial(M=cel[0], I=I0), t_final=t_final, dt=0.05)
        d_cesc = solve(cesc.with_params(**f).with_initial(M=cel[0], I=I0), t_final=t_final, dt=0.05)
        t, M_paper = np.array(d_paper['t']), np.array(d_paper['M'])
        t, M_cesc = np.array(d_cesc['t']), np.array(np.array(d_cesc['M'])*(1-np.array(d_cesc['D'])))

        M_noisy_paper = np.interp(months, t, M_paper) + rng.normal(0, noise * cel.max(), size=months.size)
        M_noisy_cesc = np.interp(months, t, M_cesc) + rng.normal(0, noise * cel.max(), size=months.size)

        ax.plot(months, cel, 'o-', color=C_CEL, mfc='white', ms=3, lw=0.8, label='CEL')
        ax.plot(months, M_noisy_paper, '-', color=C_PAPER, lw=1.1, alpha=0.75, label='M paper (noise)')
        ax.plot(months, M_noisy_cesc, '-', color=C_EXT, lw=1.1, alpha=0.75, label='M extension (noise)')
        ax.plot(t, M_paper, '--', color=C_PAPER, lw=1.8, label='M paper (no noise)')
        ax.plot(t, M_cesc, '--', color=C_EXT, lw=1.8, label='M extension (no noise)')
        style(ax, p, f, I0, t_final)
    finish(figC, axesC, 'Comparison', ncol=2)

    plt.show()


# def comparaison_8():
#     n = pd.read_whatever()
#     plt.figure()
#     for i in :

#         data = solve()
#         plt.subplot(n,1,i)
#         plt.plot()


def phi_bif(spec=CESC, t_final=600):
    phis = [0.5, 0.8, 1.5]
    fig = plt.figure(layout='constrained', figsize=(12,5))
    figA, figB = fig.subfigures(1,2)

    axA = figA.subplots()
    for phi in phis:
        data = solve(spec.with_params(phi=phi), t_final=t_final)

        t = np.array(data.get('t')) 
        M = np.array(data.get('M')) 
        D = np.array(data.get('D'))
        I = np.array(data.get('I'))

        G = M*(1 - D)
        B = 1 - G - D 
        OD = 1 - G

        peaks, _ = find_peaks(OD, prominence=0.05)        # ignora les oscil·lacions petites prop de HB
        t_last = t[peaks[-1]] if len(peaks) else np.nan

        idx = np.where(t == t_last)[0]
        y_last = OD[idx][0]  # Valor de OD en t_last
        axA.plot(t, OD, marker='o', markevery=idx, mfc='white', mec='k', ms=8)
        axA.vlines(t_last, ymin=0, ymax=y_last, ls=':', color='gray', alpha=0.5)

    axA.set_ylim(0, 1.02)
    axA.set_xlim(0, t_final)
    axA.set_xlabel('Time')
    axA.set_ylabel('OD')
    axA.set_title('Transient visualisation of the OD through time in different disease intensities')


    phis = np.linspace(0.5,2.5,60)
    t_vals = []
    y_vals = []

    for phi in phis:
        data = solve(spec.with_params(phi=phi), t_final=t_final)

        t = np.array(data.get('t')) 
        M = np.array(data.get('M')) 
        D = np.array(data.get('D'))
        OD = 1 - M*(1 - D)

        peaks, _ = find_peaks(OD, prominence=0.05)        # ignora les oscil·lacions petites prop de HB
        t_last = t[peaks[-1]] if len(peaks) else np.nan
        t_vals.append(t_last)

        idx = np.where(t == t_last)[0]
        y_last = OD[idx][0]  # Valor de OD en t_last
        y_vals.append(y_last)

    axB, axC = figB.subplots(2,1, sharex=True)
    
    axB.plot(phis, t_vals)
    axB.set_ylabel('Last oscillation time')
    axB.set_title('Last oscillation time values on phi')

    axC.plot(phis,y_vals)
    axC.set_xlabel('φ')
    axC.set_ylabel('Last remission peak value')
    axC.set_title('Last remission peak value on phi')

    plt.suptitle('Analysis of the effect of Phi on relapses')
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
        B = 1 - np.array(G) - np.array(D)

        peaks, _ = find_peaks(I, prominence=0.05)        # ignora les oscil·lacions petites prop de HB
        t_last = t[peaks[-1]] if len(peaks) else np.nan

        plt.subplot(1, 3, i+1)
        # plt.plot(t, G, label='[G]', color='#3E8A34')
        # plt.plot(t, B, label='[B]', color="#C27430")
        # plt.plot(t, D, label='[D]', color="#A71717")
        plt.plot(t, I, label='[I]', color="#A717A2")
        plt.axvline(t_last, ls=':')

        plt.title('G,B,D,I fractions through time')
        plt.xlabel('Time (months)')
        plt.ylabel('[G],[B],[D],[I]')
        plt.legend()

    plt.tight_layout()
    plt.show()


def single_trans_default(spec=CESCH, input = 'I', t_final=600, prominence=0.05):

    data = solve(spec, t_final)

    t, M, I, D = data.get('t'), data.get('M'), data.get('I'), data.get('D')

    peaks, _ = find_peaks(I, prominence=0.05)        # ignora les oscil·lacions petites prop de HB
    t_last = t[peaks[-1]] if len(peaks) else np.nan

    p = spec.params
    phi_H = p['delta'] * (1 + p['eta']) / (1 - p['eta'])
    t_hopf = p['Ta'] * np.log(p['phi'] / phi_H)       
    
    plt.figure()

    if input == 'I':
        plt.plot(t, I, label='I')
        if t_hopf > 0:
            plt.axvline(t_hopf, ls='--', c='k', lw=1, label=r'$\phi(t)=\phi_H$')
        if not np.isnan(t_last):
            plt.axvline(t_last, ls=':', c='k', lw=1, label='Last relapse')
        plt.legend()
        plt.xlim(0, t_final)


    elif input == 'OD':
        OD = np.array(M)*(1 - np.array(D))
        plt.plot(t, OD, label='OD')
        if t_hopf > 0:
            plt.axvline(t_hopf, ls='--', c='k', lw=1, label=r'$\phi(t)=\phi_H$')
        if not np.isnan(t_last):
            plt.axvline(t_last, ls=':', c='k', lw=1, label='Last relapse')
        plt.legend()
        plt.xlim(0, t_final)


def ab_discression(spec=CESCH, t_final=600, prominence=0.05, margin=60):
    data = solve(spec, t_final=t_final)
    I, t = np.array(data['I']), np.array(data['t'])
    peaks, _ = find_peaks(I, prominence=prominence)        # ignora les oscil·lacions petites prop de HB
    if len(peaks) < 2:                                     # 0-1 pics: PPMS
        return 0
    if t[peaks[-1]] < t_final - margin:                    # els brots s'aturen abans del final: RRMS+SPMS
        return 1
    return 2                                               # brots fins al final: RRMS


def ab_scatter(spec=CESCH, t_final=600, prominence=0.05, margin=60, factor=30, n=25):
    COL_RRMS_SPMS, COL_RRMS, COL_PPMS = "#E88420", "#73D635", "#A228DB"
    colors = {0: COL_PPMS, 1: COL_RRMS_SPMS, 2: COL_RRMS}
    names = {0: 'PPMS', 1: 'RRMS+SPMS', 2: 'RRMS'}

    a0, b0 = spec.params['a'], spec.params['b']
    a_vals = np.geomspace(a0 / factor, a0 * factor, n)    # espaiat logarítmic
    b_vals = np.geomspace(b0 / factor, b0 * factor, n)

    # ---------- classificació de cada punt (a, b) ----------
    A, B = np.meshgrid(a_vals, b_vals)
    canon = np.zeros(A.shape, dtype=int)
    for i in DotBar(range(n), desc='b'):
        for j in range(n):
            canon[i, j] = ab_discression(spec.with_params(a=A[i, j], b=B[i, j]),
                                         t_final, prominence, margin)

    # ---------- scatter ----------
    plt.figure(figsize=(7, 6))
    plt.scatter(A, B, c=[colors[k] for k in canon.ravel()], s=40)
    for k in names:
        plt.scatter([], [], color=colors[k], label=names[k])   # entrades de la llegenda
    plt.plot(a0, b0, 'kx', ms=10, label='default')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('a'); plt.ylabel('b'); plt.title('MS type')
    plt.grid(ls=':', color='gray')
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.tight_layout(); plt.show()
    return A, B, canon


def stackplot_RAW_PIRA(spec=CESCH, t_final=600, prominence=0.05):
    """Split of the accumulated damage D into its two sources (methods, step 3).

    A: D_RAW + D_PIRA stacked (= D).          B: rate of each contribution.
    C: myelinated (G), demyelinated (B) and dead (D) axon fractions.
    Vertical lines: phi(t) = phi_H (analytical) and last relapse (simulated).
    """
    p = spec.params
    phi_H = p['delta'] * (1 + p['eta']) / (1 - p['eta'])
    t_hopf = p['Ta'] * np.log(p['phi'] / phi_H)            # mesos; < 0 si phi0 < phi_H
    C_RAW, C_PIRA = '#d95f02', '#7570b3'

    d = solve(spec, t_final=t_final)
    t = np.array(d['t'])
    ty = t / 12                                            # mesos -> anys
    M, I, D = np.array(d['M']), np.array(d['I']), np.array(d['D'])
    RAW, PIRA = np.array(d['RAW']), np.array(d['PIRA'])

    # ---------- últim brot: últim pic de I prou gran ----------
    peaks, _ = find_peaks(I, prominence=prominence)        # ignora les oscil·lacions petites prop de HB
    t_last = t[peaks[-1]] if len(peaks) else np.nan

    # A i B a dalt, C ocupa tota la fila de baix
    fig, axs = plt.subplot_mosaic([['A', 'B'], ['C', 'C']], figsize=(13, 10))
    axA, axB, axC = axs['A'], axs['B'], axs['C']

    # ---------- A: D = D_RAW + D_PIRA ----------
    axA.stackplot(ty, RAW, PIRA, colors=[C_RAW, C_PIRA], alpha=0.8,
                  labels=['Demyelination-driven (a)', 'Background (b)'])
    axA.set_ylabel('D')
    axA.set_title('Accumulated damage split by source')

    # ---------- B: velocitat de cada contribució ----------
    axB.plot(ty, d['dRAW'], color=C_RAW, label='dRAW/dt')
    axB.plot(ty, d['dPIRA'], color=C_PIRA, label='dPIRA/dt')
    axB.set_ylabel('dD/dt contribution')
    axB.set_title('Damage rate of each source')

    # ---------- C: axons mielinitzats (G), desmielinitzats (B) i morts (D) ----------
    G = M * (1 - D)                                        # G + B + D = 1
    B = 1 - G - D
    axC.plot(ty, G, label='Myelinated (G)', color='#053E80')
    axC.plot(ty, B, label='Demyelinated (B)', color='#75C8C0')
    axC.plot(ty, D, label='Dead axons (D)', color='#3E8A34')
    axC.set_ylabel('G, B, D')
    axC.set_title('Myelin, Demyelin, Dead axons')

    # ---------- format comú: línies de referència, eixos, lletres ----------
    for letter, ax in axs.items():
        if t_hopf > 0:
            ax.axvline(t_hopf / 12, ls='--', c='k', lw=1, label=r'$\phi(t)=\phi_H$')
        if not np.isnan(t_last):
            ax.axvline(t_last / 12, ls=':', c='k', lw=1, label='Last relapse')
        ax.set_xlim(0, ty[-1])
        ax.set_xlabel('Time (years)')
        ax.grid(ls=':', color='gray')
        ax.legend(loc='upper left', fontsize=8)
        ax.text(0.97, 0.95, letter, transform=ax.transAxes, ha='right', va='top', fontsize=16)

    plt.tight_layout(); plt.show()

def main():
#------------------------------ PAPER --------------------------------
    # figure2()
    # figure3()
    # figure4()
    # figure6(n=30, dt=0.1, t_final=2000)
    # figure7()

#------------------------------  CESC  --------------------------------
    # transient(CESC.with_params(a=0.01,b=0.01))
    # single_trans_default(CESCH.with_params(a=0, b=0.05), prominence=0.05)
    # comparaison_7()
    # stackplot_RAW_PIRA(CESCH.with_params(phi=0.7))
    # ab_scatter()
    phi_bif()
    # single_trans_default(CESC.with_params(phi=3), input='OD')
    # plt.show()




if __name__ == '__main__':
    main()
