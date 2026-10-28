from cModelSpec import ModelSpec

def solve(spec: ModelSpec, t_final=300.0, dt=0.05, solver='RK4') -> dict:
        m = spec.build()
        result = {'t': []}
        result |= {k: [] for k in spec.state_vars}
        result |= {'d' + k: [] for k in spec.state_vars}
        result |= {k: [] for k in spec.probes}

        for _ in range(int(round(t_final / dt))):
            m.execute_step(dt=dt, solver=solver)
            result['t'].append(m.time.value)
            for k in spec.state_vars:
                result[k].append(m.variables[k].value)
                result['d' + k].append(m.variables[k].derivative)
            for k in spec.probes:
                result[k].append(m.variables[k].value)
        return result

