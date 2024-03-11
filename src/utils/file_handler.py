

from datetime import datetime
import json
from pathlib import Path

import numpy as np


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: object):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return o.__str__()
        if isinstance(o, datetime):
            return o.__str__()
        return super().default(o)