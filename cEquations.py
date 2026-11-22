from cModelSpec import ModelSpec
from cBareBones import (Op_get, Op_set_der, Op_prod, Op_sign, Op_sum, Op_subs,
                        Op_div, Op_const, Op_pow, Op_exp, Op_time)
from cStability import build_jacobian, build_steady_states


# =============================================================================
# PAPER:  dM/dt = r*M*(1 - M) - phi*I*M/(M + eta)
#         dI/dt = phi*I*M/(M + eta) - delta*I
# =============================================================================

def build_paper_step_list(m) -> list:
    return [
        #  M equation:  dM/dt = r*M*(1 - M) - phi*I*M/(M + eta)
        Op_get(m, name='r'),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_const(m, value=1),
        Op_get(m, name='M'),
        Op_sign(m),
        Op_sum(m),
        Op_prod(m),

        Op_get(m, name='phi'), # ------ Numerator ---------
        Op_get(m, name='I'),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_prod(m),

        Op_get(m, name='M'), # ------ Denominator --------
        Op_get(m, name='eta'),
        Op_sum(m),
        Op_div(m),

        Op_sign(m),
        Op_sum(m),
        Op_set_der(m, name='M'),

        #  I equation:  dI/dt = phi*I*M/(M + eta) - delta*I
        Op_get(m, name='phi'), # ------ Numerator ---------
        Op_get(m, name='I'),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_prod(m),

        Op_get(m, name='M'), # ------ Denominator --------
        Op_get(m, name='eta'),
        Op_sum(m),
        Op_div(m),

        Op_get(m, name='delta'),
        Op_get(m, name='I'),
        Op_prod(m),

        Op_sign(m),
        Op_sum(m),

        Op_set_der(m, name='I')
    ]


def paper_jacobian(state: list, p: dict) -> list:
    """Jacobian matrix of the paper model at state = [M, I]."""
    M, I = state
    r, phi, eta, delta = p['r'], p['phi'], p['eta'], p['delta']
    return [
        [r*(1 - 2*M) - phi*eta*I/(M + eta)**2,   -phi*M/(M + eta)],
        [phi*eta*I/(M + eta)**2,                  phi*M/(M + eta) - delta],
    ]


def paper_steady_states(p: dict) -> dict:
    """Equilibria of the paper model, as [M, I]."""
    r, phi, eta, delta = p['r'], p['phi'], p['eta'], p['delta']
    return {
        '0,0': [0, 0],                                   # M=0, I=0
        '1,0': [1, 0],                                   # M=1, I=0
        '*,*': [delta*eta/(phi - delta),
                r*eta*(phi - delta - delta*eta)/(phi - delta)**2],
    }


PAPER = ModelSpec(
    name='paper',
    state_vars={'M': 1, 'I': 0.2},
    params={'r': 0.5, 'phi': 0.45, 'eta': 0.5, 'delta': 0.2},
    build_steps=build_paper_step_list,
    jacobian=build_jacobian(paper_jacobian),
    steady_states=build_steady_states(paper_steady_states),
)


# =============================================================================
# CESC:   paper model + accumulated damage D (extension, eqs. 2-3)
#         dM/dt = r(D)*M*(1 - M) - phi(t)*I*M/(M + eta)
#         dI/dt = phi(t)*I*M/(M + eta) - delta*I
#         dD/dt = [a*(1 - M) + b]*(1 - D)
#         r(D)   = r0*(1 - D)**c        (remyelination fails with damage)
#         phi(t) = phi0*exp(-t/Ta)      (inflammatory drive declines with age)
#         t in months -> Ta = 50 years = 600 months
# =============================================================================

def build_cesc_step_list(m) -> list:
    return [
        #  M equation:  dM/dt = r0*(1 - D)**c * M*(1 - M) - phi(t)*I*M/(M + eta)
        Op_get(m, name='r'),       # ------ r(D) = r0*(1 - D)**c ------
        Op_const(m, value=1),
        Op_get(m, name='D'),
        Op_subs(m),
        Op_get(m, name='c'),
        Op_pow(m),
        Op_prod(m),

        Op_get(m, name='M'),        # ------ * M*(1 - M) ------
        Op_prod(m),
        Op_const(m, value=1),
        Op_get(m, name='M'),
        Op_subs(m),
        Op_prod(m),

        Op_get(m, name='phi'),     # ------ phi(t) = phi0*exp(-t/Ta) ------
        Op_time(m),
        Op_get(m, name='Ta'),
        Op_div(m),
        Op_sign(m),
        Op_exp(m),
        Op_prod(m),

        Op_get(m, name='I'),        # ------ * I*M/(M + eta) ------
        Op_prod(m),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_get(m, name='M'),
        Op_get(m, name='eta'),
        Op_sum(m),
        Op_div(m),

        Op_subs(m),
        Op_set_der(m, name='M'),

        #  I equation:  dI/dt = phi(t)*I*M/(M + eta) - delta*I
        Op_get(m, name='phi'),     # ------ phi(t) = phi0*exp(-t/Ta) ------
        Op_time(m),
        Op_get(m, name='Ta'),
        Op_div(m),
        Op_sign(m),
        Op_exp(m),
        Op_prod(m),

        Op_get(m, name='I'),        # ------ * I*M/(M + eta) ------
        Op_prod(m),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_get(m, name='M'),
        Op_get(m, name='eta'),
        Op_sum(m),
        Op_div(m),

        Op_get(m, name='delta'),    # ------ - delta*I ------
        Op_get(m, name='I'),
        Op_prod(m),
        Op_subs(m),
        Op_set_der(m, name='I'),

        # D equation: dD/dt = [a*(1 - M) + b]*(1 - D)
        Op_get(m, name='a'),
        Op_const(m, value=1),
        Op_get(m, name='M'),
        Op_subs(m),
        Op_prod(m),
        Op_get(m, name='b'),
        Op_sum(m),
        Op_const(m, value=1),
        Op_get(m, name='D'),
        Op_subs(m),
        Op_prod(m),
        Op_set_der(m, name='D')
    ]


# Steady states and stability: slow-fast analysis (methods, step 2).
# D and phi change slowly, so they are frozen and the fast subsystem (M, I)
# is analysed:  D = D_frozen,  phi = phi0  (sweep 'phi0' or 'D_frozen').

def cesc_jacobian(state: list, p: dict) -> list:
    """Jacobian of the fast subsystem (M, I) at state = [M, I], with D and phi frozen."""
    M, I = state
    phi, eta, delta = p['phi0'], p['eta'], p['delta']
    r = p['r0'] * (1 - p['D_frozen'])**p['c']
    return [
        [r*(1 - 2*M) - phi*eta*I/(M + eta)**2,   -phi*M/(M + eta)],
        [phi*eta*I/(M + eta)**2,                  phi*M/(M + eta) - delta],
    ]


def cesc_steady_states(p: dict) -> dict:
    """Equilibria of the fast subsystem (M, I), with D and phi frozen."""
    phi, eta, delta = p['phi0'], p['eta'], p['delta']
    r = p['r0'] * (1 - p['D_frozen'])**p['c']
    return {
        '0,0': [0, 0],                                   # M=0, I=0
        '1,0': [1, 0],                                   # M=1, I=0
        '*,*': [delta*eta/(phi - delta),
                r*eta*(phi - delta - delta*eta)/(phi - delta)**2],
    }


CESC = ModelSpec(
    name='cesc',
    state_vars={'M': 1.0, 'I': 0.2, 'D': 0.0},
    params={'r': 0.5, 'c': 1.0, 'phi': 0.9, 'Ta': 600.0, 'eta': 0.5, 'delta': 0.2,
            'a': 0.0018, 'b': 0.0017,
            'D_frozen': 0.0},           # only for the steady-state analysis
    build_steps=build_cesc_step_list,
    jacobian=build_jacobian(cesc_jacobian),
    steady_states=build_steady_states(cesc_steady_states),
)


# =============================================================================
# CESCh:   paper model + accumulated damage D (extension, eqs. 2-3) +  helpers (RAW + PIR)
#         dM/dt = r(D)*M*(1 - M) - phi(t)*I*M/(M + eta)
#         dI/dt = phi(t)*I*M/(M + eta) - delta*I
#         dD/dt = [a*(1 - M) + b]*(1 - D)
#         r(D)   = r0*(1 - D)**c        (remyelination fails with damage)
#         phi(t) = phi0*exp(-t/Ta)      (inflammatory drive declines with age)
#         t in months -> Ta = 50 years = 600 months
# =============================================================================

def build_cesch_step_list(m) -> list:
    return [
        #  M equation:  dM/dt = r0*(1 - D)**c * M*(1 - M) - phi(t)*I*M/(M + eta)
        Op_get(m, name='r'),       # ------ r(D) = r0*(1 - D)**c ------
        Op_const(m, value=1),
        Op_get(m, name='D'),
        Op_subs(m),
        Op_get(m, name='c'),
        Op_pow(m),
        Op_prod(m),

        Op_get(m, name='M'),        # ------ * M*(1 - M) ------
        Op_prod(m),
        Op_const(m, value=1),
        Op_get(m, name='M'),
        Op_subs(m),
        Op_prod(m),

        Op_get(m, name='phi'),     # ------ phi(t) = phi0*exp(-t/Ta) ------
        Op_time(m),
        Op_get(m, name='Ta'),
        Op_div(m),
        Op_sign(m),
        Op_exp(m),
        Op_prod(m),

        Op_get(m, name='I'),        # ------ * I*M/(M + eta) ------
        Op_prod(m),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_get(m, name='M'),
        Op_get(m, name='eta'),
        Op_sum(m),
        Op_div(m),

        Op_subs(m),
        Op_set_der(m, name='M'),

        #  I equation:  dI/dt = phi(t)*I*M/(M + eta) - delta*I
        Op_get(m, name='phi'),     # ------ phi(t) = phi0*exp(-t/Ta) ------
        Op_time(m),
        Op_get(m, name='Ta'),
        Op_div(m),
        Op_sign(m),
        Op_exp(m),
        Op_prod(m),

        Op_get(m, name='I'),        # ------ * I*M/(M + eta) ------
        Op_prod(m),
        Op_get(m, name='M'),
        Op_prod(m),
        Op_get(m, name='M'),
        Op_get(m, name='eta'),
        Op_sum(m),
        Op_div(m),

        Op_get(m, name='delta'),    # ------ - delta*I ------
        Op_get(m, name='I'),
        Op_prod(m),
        Op_subs(m),
        Op_set_der(m, name='I'),

        # D equation: dD/dt = [a*(1 - M) + b]*(1 - D)
        Op_get(m, name='a'),
        Op_const(m, value=1),
        Op_get(m, name='M'),
        Op_subs(m),
        Op_prod(m),
        Op_get(m, name='b'),
        Op_sum(m),
        Op_const(m, value=1),
        Op_get(m, name='D'),
        Op_subs(m),
        Op_prod(m),
        Op_set_der(m, name='D'),

        # RAW: a*(1 - M)*(1 - D)
        Op_get(m, name='a'),
        Op_const(m, value=1), Op_get(m, name='M'), Op_subs(m),
        Op_prod(m),
        Op_const(m, value=1), Op_get(m, name='D'), Op_subs(m),
        Op_prod(m),
        Op_set_der(m, name='RAW'),

        # PIRA: b*(1 - D)
        Op_get(m, name='b'),
        Op_const(m, value=1), Op_get(m, name='D'), Op_subs(m),
        Op_prod(m),
        Op_set_der(m, name='PIRA')
    ]

def cesch_jacobian(state: list, p: dict) -> list:
    """Jacobian of the fast subsystem (M, I) at state = [M, I], with D and phi frozen."""
    M, I = state
    phi, eta, delta = p['phi'], p['eta'], p['delta']
    r = p['r'] * (1 - p['D_frozen'])**p['c']
    return [
        [r*(1 - 2*M) - phi*eta*I/(M + eta)**2,   -phi*M/(M + eta)],
        [phi*eta*I/(M + eta)**2,                  phi*M/(M + eta) - delta],
    ]


def cesch_steady_states(p: dict) -> dict:
    """Equilibria of the fast subsystem (M, I), with D and phi frozen."""
    phi, eta, delta = p['phi'], p['eta'], p['delta']
    r = p['r'] * (1 - p['D_frozen'])**p['c']
    return {
        '0,0': [0, 0],                                   # M=0, I=0
        '1,0': [1, 0],                                   # M=1, I=0
        '*,*': [delta*eta/(phi - delta),
                r*eta*(phi - delta - delta*eta)/(phi - delta)**2],
    }


CESCH = ModelSpec(
    name='cesc_h',
    state_vars={'M': 1.0, 'I': 0.2, 'D': 0.0, 'RAW': 0.0, 'PIRA': 0.0},
    params={'r': 0.5, 'c': 1.0, 'phi': 0.9, 'Ta': 600.0, 'eta': 0.5, 'delta': 0.2,
            'a': 0.0018, 'b': 0.0017,
            'D_frozen': 0.0},           # only for the steady-state analysis
    build_steps=build_cesch_step_list,
    jacobian=build_jacobian(cesch_jacobian),
    steady_states=build_steady_states(cesch_steady_states),
)