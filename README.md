# Biosist_model7

Minimal ODE model of multiple sclerosis (myelin `M`, inflammation `I`), reproducing Jenner, Weatherley & Frascoli, and an extension with accumulated irreversible damage `D` to model the transition from relapsing–remitting to progressive disease.

Systems Biology Lab Projects, Universitat Pompeu Fabra.

## Code structure

The code is split into three layers: a small numerical engine, the model definitions, and the analysis/plotting scripts.

```
main.py          Figures (entry point)
Solve.py         solve() and steady_state() drivers
cEquations.py    Model definitions: PAPER and CESC
cModelSpec.py    ModelSpec: immutable description of a model
cStability.py    Generic Jacobian / steady-state helpers
cModel.py        Numerical engine: variables, parameters, Euler & RK4
cBareBones.py    Stack operators the equations are written with
cUtilities.py    Loader for the patient data (XLS)
```

### 1. Numerical engine (`cModel.py`, `cBareBones.py`)

- Each model's equations are compiled into `Model.step_list`, a flat list of **stack operators** in reverse Polish notation (`Op_get`, `Op_prod`, `Op_sum`, `Op_div`, `Op_exp`, `Op_time`, …).
- Running the list once writes the derivative of every state variable with `Op_set_der`.
- `Model` stores state variables, parameters and probes, and integrates them with `Euler` or `RK4`. The solvers only call the operator list, so they don't depend on the model.
- `model.repr_steps()` prints the compiled operations for debugging.

### 2. Model definitions (`cModelSpec.py`, `cEquations.py`, `cStability.py`)

- `ModelSpec` is a frozen dataclass that bundles a model's initial state, parameters, equation builder, Jacobian and steady states. It checks on creation that every name the equations use is defined and that every state variable has an equation.
- `spec.with_params(...)` and `spec.with_initial(...)` return modified copies, e.g. `PAPER.with_params(phi=0.7)`.
- `cEquations.py` defines two models:
  - **`PAPER`**: the original two-equation model, with its analytical equilibria and Jacobian.
    ```
    dM/dt = r M (1 − M) − φ M I / (M + η)
    dI/dt = φ M I / (M + η) − δ I
    ```
  - **`CESC`**: the extension. It adds damage `D`, damage-dependent remyelination `r(D) = r0 (1 − D)^c` and a declining drive `φ(t) = φ0 e^(−t/Ta)`.
    ```
    dD/dt = [a (1 − M) + b] (1 − D)
    ```
    Its stability analysis uses a slow–fast approach: `D` and `φ` are frozen as `D_frozen` and `phi0`, and the fast `(M, I)` subsystem is analysed.
- `cStability.py` turns each model's Jacobian and steady-state definitions into the callables `ModelSpec` expects. It classifies stability from the eigenvalues.

### 3. Analysis and figures (`Solve.py`, `main.py`)

- `solve(spec, t_final, dt, solver)` integrates a model. It returns a dict of time series: `t`, each variable, its derivative `d<var>`, and the probes.
- `steady_state(spec, param_range, param_name)` sweeps one parameter and returns every equilibrium together with its stability.
- `main.py` builds the figures:

  | Function | Content |
  |---|---|
  | `figure2()` | Time courses of `M`, `I` for three values of φ |
  | `figure3()` | Bifurcation diagram in φ (BP, Hopf, limit-cycle maxima, period inset) |
  | `figure4()` | Bifurcation diagram in η and normalised orbits |
  | `figure7()` | Patient data (CEL, EDSS, relapses) and model fits |
  | `transient()` | Extended model: `M`, `I`, `D` and total myelin `G = M (1 − D)` |

  Use `transient(debug=True)` to recover the original model: it sets `a = b = 0` and makes φ constant (`Ta = 10^12`).

### Data and output

- `12918_2010_719_MOESM3_ESM.XLS`: monthly MRI/EDSS data for 9 patients over 48 months (Vélez de Mendizábal et al., 2011, Additional file 3). Only `figure7()` uses it.
- `figures/`: saved figures. 

## Running

Requires Python 3.10+ with `numpy`, `matplotlib`, `pandas` and `xlrd` (`xlrd` is needed to read the `.XLS` file).

```bash
pip install numpy matplotlib pandas xlrd
python main.py
```

Choose which figures to draw in `main()`. `load_patients()` looks for the XLS at `./Biosist_model7/<file>`, so run `figure7()` from the folder that contains the repository.

## License

Apache 2.0.