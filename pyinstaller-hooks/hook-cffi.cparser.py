"""Local PyInstaller hook for cffi.cparser.

The module contains a never-called helper whose only purpose is working
around static import finders by importing pycparser's generated tables.
Those files are not shipped by all pycparser wheels, so excluding them
from analysis avoids false-positive missing-module warnings.
"""

excludedimports = ["pycparser.lextab", "pycparser.yacctab"]
