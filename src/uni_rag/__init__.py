"""Local-only university document retrieval."""

import os

# Set before importing optional dependencies. No remote models/tokenizers are used.
for _name, _value in {
    "DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "ANONYMIZED_TELEMETRY": "False",
}.items():
    os.environ[_name] = _value
