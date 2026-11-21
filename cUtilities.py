from collections import defaultdict
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_patients(path='12918_2010_719_MOESM3_ESM.XLS'):
    """Dades de Vélez de Mendizábal et al. 2011 (Additional file 3), 9 pacients
    amb RM mensual durant 48 mesos. Retorna {pacient: {'CEL', 'EDSS', 'RELAPSES'}}."""
    raw = pd.read_excel(Path.cwd() / 'Biosist_model7' / path, sheet_name='data', header=None)
    data = defaultdict(dict)
    for _, row in raw.iterrows():
        label = str(row[0])
        for key in ('CEL', 'EDSS', 'RELAPSES'):
            if label.startswith(key + '#'):
                p = int(label.split('#')[1])
                data[p][key] = row[1:49].astype(float).to_numpy()
    return dict(data)


class DotBar(tqdm):
    """tqdm progress bar drawn as '█ █ █ · · ·' with a fixed number of blocks."""

    def __init__(self, *args, blocks: int = 12, **kwargs):
        self.blocks = blocks
        kwargs.setdefault('bar_format',
                          '{desc}: {percentage:3.0f}% |{dots}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict
        filled = int(self.blocks * d['n'] / d['total']) if d['total'] else 0
        d['dots'] = ' '.join('█' * filled + '·' * (self.blocks - filled))
        return d
