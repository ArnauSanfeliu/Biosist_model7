"""Stack operators: the instructions a compiled circuit is made of.

Each operator pops its operands from the stack and pushes the result, in
reverse Polish notation. `Op_get` reads a variable or parameter by name,
`Op_set_der` writes the final value of a derivative.
"""

import math

from cModel import BaseOperator, Vals, Param

class Delay(BaseOperator):
    name: str
    def __init__(self, model, name: str):
        super().__init__(model)
        self.name = name

    def execute(self, stack: list):
        self.model.variables[self.name].new_value = stack.pop()

# ============================

class Operator(BaseOperator):
    def __init__(self, model):
        super().__init__(model)

class Op_get(Operator):
    name: str
    def __init__(self, model, name: str):
        super().__init__(model)
        self.name = name

    def execute(self, stack: list):
        stack.append(self.model.variables[self.name].value)

class Op_get_der(Operator):
    name: str
    def __init__(self, model, name: str):
        super().__init__(model)
        self.name = name

    def execute(self, stack: list):
        stack.append(self.model.variables[self.name].derivative)

class Op_set(Operator):
    name: str
    def __init__(self, model, name: str):
        super().__init__(model)
        self.name = name

    def execute(self, stack: list):
        self.model.variables[self.name].value = stack.pop()

class Op_set_der(Operator):
    name: str
    def __init__(self, model, name: str):
        super().__init__(model)
        self.name = name

    def execute(self, stack: list):
        self.model.variables[self.name].derivative = stack.pop()

class Op_const(Operator):
    def __init__(self, model, value: float | int):
        super().__init__(model)
        self.value = value

    def execute(self, stack: list):
        stack.append(self.value)

class Op_sum(Operator):
    def __init__(self, model):
        super().__init__(model)

    def execute(self, stack: list):
        var2 = stack.pop()
        var1 = stack.pop()
        stack.append(var1+var2)

class Op_sign(Operator):
    def __init__(self, model):
        super().__init__(model)

    def execute(self, stack: list):
        var = stack.pop()
        stack.append(-var)

class Op_subs(Operator):
    def __init__(self, model):
        super().__init__(model)

    def execute(self, stack: list):
        var2 = stack.pop()
        var1 = stack.pop()
        stack.append(var1-var2)

class Op_prod(Operator):
    def __init__(self, model):
        super().__init__(model)

    def execute(self, stack: list):
        var2 = stack.pop()
        var1 = stack.pop()
        stack.append(var1*var2)

class Op_div(Operator):
    def __init__(self, model):
        super().__init__(model)

    def execute(self, stack: list):
        var2 = stack.pop()
        var1 = stack.pop()
        stack.append(var1/var2)

class Op_pow(Operator):
    def __init__(self, model):
        super().__init__(model)

    def execute(self, stack: list):
        var2 = stack.pop()
        var1 = stack.pop()
        stack.append(var1**var2)


#=======================================================
# DEBUGGING AND PROBES
#=======================================================

class Op_print(Operator):
    def __init__(self, model):
        super().__init__(model)

    def execute(self, stack: list):
        var = stack[-1]
        print(var)

class Op_probe(Operator):
    """Copies the top of the stack (the term just computed) into a probe."""
    name: str
    def __init__(self, model, name: str):
        super().__init__(model)
        self.name = name

    def execute(self, stack: list):
        self.model.variables[self.name].value = stack[-1]


#======================================================
# TIME-DEPENDENT INPUTS (extracellular supplies)
#======================================================

class Op_step(Operator):
    """Pushes 1 inside the window t_on <= t < t_off, 0 outside."""
    def __init__(self, model, t_on: str, t_off: str):
        super().__init__(model)
        self.t_on, self.t_off = t_on, t_off

    def execute(self, stack: list):
        t = self.model.variables['t'].value
        on = self.model.variables[self.t_on].value
        off = self.model.variables[self.t_off].value
        stack.append(1.0 if on <= t < off else 0.0)

class Op_sin(Operator):
    """Pushes (1 + sin(2*pi*t/T_period)) / sin_range.

    The factor oscillates between 0 and 2/sin_range, so it never makes a
    supply negative."""
    def __init__(self, model, T_period, sin_range):
        super().__init__(model)
        self.T_period = T_period
        self.sin_range = sin_range

    def execute(self, stack):
        T_period = self.model.variables[self.T_period].value
        sin_range = self.model.variables[self.sin_range].value
        t = self.model.variables['t'].value
        val = (1 + math.sin(2*math.pi*t/T_period))/sin_range
        stack.append(val)

class Op_exp(Operator):
    """Pops x and pushes exp(x)."""
    def __init__(self, model):
        super().__init__(model)

    def execute(self, stack):
        stack.append(math.exp(stack.pop()))

#======================================================
# RAPID-EQUILIBRIUM BINDING
#======================================================

def _increasing_root(f, df, lo, hi, x):
    """Root of f, increasing on [lo, hi] with f(lo) <= 0 <= f(hi).

    Safeguarded Newton: a step that leaves the bracket (or a useless
    derivative) is replaced by bisection. The bracket only shrinks, so it
    always converges; Newton keeps the number of steps small. It stops when
    floating point cannot improve the result any more.
    """
    for _ in range(200):
        fx, dfx = f(x), df(x)
        if fx > 0.0:
            hi = x
        elif fx < 0.0:
            lo = x
        else:
            return x
        new = x - fx / dfx if dfx and dfx > 0.0 else None
        if new is None or not (lo < new < hi):
            new = 0.5 * (lo + hi)
        if new == x or new <= lo or new >= hi:
            break
        x = new
    return x


def solve_complex(at, bt, n1, n2, kd, guess=None):
    """Rapid binding equilibrium. Returns (C, free A, free B).

        n1 A + n2 B <-> C,     KD = A^n1 * B^n2 / C
        A_T = A + n1*C,        B_T = B + n2*C

    `at`, `bt` are the totals; `bt=None` means a single reactant
    (n1 A <-> C, homomultimer) and the returned B is None.

    C is the only root of

        g(C) = KD*C - (A_T - n1*C)^n1 * (B_T - n2*C)^n2

    g is increasing, negative at C = 0 and positive at Cmax = min(A_T/n1,
    B_T/n2), and the root is found with `_increasing_root`.

    Precision: when the limiting reactant is almost fully bound (C ~ Cmax),
    its free form B_T - n2*C is a difference of two nearly equal numbers and
    loses all its digits, which is exactly the regime that matters in
    titration. In that case the problem is solved again for d = Cmax - C,
    so the free limiting species is n*d with no subtraction. All free forms
    are returned ready, so the caller never repeats the subtraction.
    """
    at = max(0.0, at)
    single = bt is None
    if not single:
        bt = max(0.0, bt)
    if n1 <= 0 or (not single and n2 <= 0) or at <= 0.0 \
            or (not single and bt <= 0.0):
        return 0.0, at, (None if single else bt)
    if kd <= 0.0:                         # infinitely tight binding
        c = at / n1 if single else min(at / n1, bt / n2)
        a = 0.0 if single or at / n1 <= bt / n2 else at - n1 * c
        b = None if single else (0.0 if bt / n2 <= at / n1 else bt - n2 * c)
        return c, a, b

    # --- Exact closed forms (no dangerous subtractions) ---
    if single and n1 == 1:
        return at / (1.0 + kd), at * kd / (1.0 + kd), None
    if single and n1 == 2:
        a = 2.0 * at / (1.0 + math.sqrt(1.0 + 8.0 * at / kd))
        return a * a / kd, a, None

    # --- General case. Unknown: C ---
    lim_is_a = single or at / n1 <= bt / n2
    cmax = at / n1 if lim_is_a else bt / n2

    def free(c):
        xa = at - n1 * c
        xb = None if single else bt - n2 * c
        return xa, xb

    def prod(xa, xb):
        p = max(0.0, xa) ** n1
        if xb is not None:
            p *= max(0.0, xb) ** n2
        return p

    def dprod(p, xa, xb):
        # d/dx of (A^n1 B^n2) when A and B grow as n1*x and n2*x
        if xa <= 0.0 or (xb is not None and xb <= 0.0):
            return 0.0
        return p * (n1 * n1 / xa + (0.0 if xb is None else n2 * n2 / xb))

    def g(c):
        return kd * c - prod(*free(c))

    def dg(c):
        xa, xb = free(c)
        return kd + dprod(prod(xa, xb), xa, xb)

    x0 = guess if (guess is not None and 0.0 < guess < cmax) else 0.5 * cmax
    c = _increasing_root(g, dg, 0.0, cmax, x0)
    if c <= 0.5 * cmax:
        xa, xb = free(c)
        return c, max(0.0, xa), (None if single else max(0.0, xb))

    # --- Limiting reactant nearly used up. Unknown: d = Cmax - C ---
    # Free limiting species = n*d (exact); the other one is what is left
    # with the complex saturated (a constant) plus n*d.
    if lim_is_a:
        rest_b = None if single else max(0.0, bt - n2 * cmax)
    else:
        rest_a = max(0.0, at - n1 * cmax)

    def free_d(d):
        if lim_is_a:
            return n1 * d, (None if single else rest_b + n2 * d)
        return rest_a + n1 * d, n2 * d

    def h(d):                              # = -g(Cmax - d), increasing in d
        return prod(*free_d(d)) - kd * (cmax - d)

    def dh(d):
        xa, xb = free_d(d)
        return kd + dprod(prod(xa, xb), xa, xb)

    d = _increasing_root(h, dh, 0.0, cmax, cmax - c)
    xa, xb = free_d(d)
    return cmax - d, xa, xb


class Op_bind_eq(Operator):
    """Solves one rapid equilibrium and writes C and the free forms.

    Does not touch the stack. It must come FIRST in step_list, before any
    derivative, so every block that reads A, B or C sees this sub-step's
    values.

        a_tot, b_tot  state variables (the totals, which are integrated)
        a, b          algebraic variables: the free forms
        c             algebraic variable: the complex
        n1, n2, kd    parameter names (read on every call, because the
                      result sliders change them live)

    `b` and `b_tot` are None when only one reactant is connected.
    """

    def __init__(self, model, a, a_tot, b, b_tot, c, n1, n2, kd):
        super().__init__(model)
        self.a, self.a_tot = a, a_tot
        self.b, self.b_tot = b, b_tot
        self.c = c
        self.n1, self.n2, self.kd = n1, n2, kd
        self._guess = None                 # Newton warm start

    def execute(self, stack):
        v = self.model.variables
        at = v[self.a_tot].value
        n1 = v[self.n1].value
        kd = v[self.kd].value
        if self.b is not None:
            bt = v[self.b_tot].value
            n2 = v[self.n2].value
        else:
            bt, n2 = None, 1.0
        c, a, b = solve_complex(at, bt, n1, n2, kd, self._guess)
        self._guess = c
        v[self.c].value = c
        v[self.a].value = a
        if self.b is not None:
            v[self.b].value = b
