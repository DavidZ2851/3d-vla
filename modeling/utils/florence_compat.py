"""Let Florence-2's remote code load without flash_attn installed.

Its modeling file lists flash_attn among its imports, so transformers' check_imports refuses to load
it when the package is missing, although flash attention is only used when explicitly requested.
Dropping it from the import list (the usual workaround) lets Florence-2 load with SDPA/eager attention.
"""

import importlib.util

import transformers.dynamic_module_utils as _dmu

if importlib.util.find_spec("flash_attn") is None and not getattr(_dmu, "_florence_compat", False):
    _get_imports = _dmu.get_imports

    def _get_imports_without_flash_attn(filename):
        return [imp for imp in _get_imports(filename) if imp != "flash_attn"]

    _dmu.get_imports = _get_imports_without_flash_attn
    _dmu._florence_compat = True
