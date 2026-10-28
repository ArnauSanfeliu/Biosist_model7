from dataclasses import dataclass, field, replace
from typing import Callable
from cModel import Model

@dataclass(frozen=True)
class ModelSpec:
    state_vars: dict[str, float]          # name -> initial value
    params: dict[str, float]              # name -> value
    build_steps: Callable[[Model], list]  
    probes: dict[str, float] = field(default_factory=dict)
    name: str = ''

    def __post_init__(self):
        self.validate()

    def build(self) -> Model:
        m = Model()
        m.name = self.name
        for k,v in self.state_vars.items(): m.add_var(k, v)
        for k,v in self.params.items():     m.add_param(k, v)
        for k,v in self.probes.items():     m.add_probe(k, v)
        m.finalize()
        m.step_list = self.build_steps(m)
        return m


    # ---------------------------------------------------------------------------
    

    def with_params(self, **changes) -> 'ModelSpec':
        """Copy of the spec with some parameters changed.

        Example: PAPER.with_params(phi=0.7, r=0.3)
        """
        unknown = changes.keys() - self.params.keys()
        if unknown:
            raise KeyError(f'Unknown parameters: {sorted(unknown)}. '
                           f'Available: {sorted(self.params)}')
        return replace(self, params=self.params | changes)

    def with_initial(self, **changes) -> 'ModelSpec':
        """Copy of the spec with some initial values changed.

        Example: PAPER.with_initial(M=0.1, I=0.5)
        """
        unknown = changes.keys() - self.state_vars.keys()
        if unknown:
            raise KeyError(f'Unknown state variables: {sorted(unknown)}. '
                           f'Available: {sorted(self.state_vars)}')
        return replace(self, state_vars=self.state_vars | changes)

    def validate(self):
        """Check that the names and the equations match. Runs automatically
        when the spec is created (from __post_init__)."""
        from cBareBones import Op_set_der    # local import: avoids a circular import

        # 1. No name defined in more than one group
        groups = {'state_vars': self.state_vars,
                  'params': self.params,
                  'probes': self.probes}
        seen = {}
        for group, names in groups.items():
            for k in names:
                if k in seen:
                    raise ValueError(f"'{k}' is defined in both {seen[k]} and {group}")
                seen[k] = group

        # 2. Every name used by the equations is defined
        steps = self.build().step_list
        used = {op.name for op in steps if hasattr(op, 'name')}
        missing = used - seen.keys()
        if missing:
            raise ValueError(f'Equations use undefined names: {sorted(missing)}')

        # 3. Every state variable has an equation
        with_eq = {op.name for op in steps if isinstance(op, Op_set_der)}
        no_eq = self.state_vars.keys() - with_eq
        if no_eq:
            raise ValueError(f'State variables with no equation: {sorted(no_eq)}')
