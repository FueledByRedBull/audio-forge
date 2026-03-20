from PyInstaller.utils.hooks import can_import_module, is_module_satisfies

hiddenimports = ["scipy.special._ufuncs_cxx"]

if is_module_satisfies("scipy >= 1.13.0") and can_import_module("scipy.special._cdflib"):
    hiddenimports += ["scipy.special._cdflib"]

if is_module_satisfies("scipy >= 1.14.0") and can_import_module("scipy.special._special_ufuncs"):
    hiddenimports += ["scipy.special._special_ufuncs"]
