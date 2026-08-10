from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("course_large_array_data")
except PackageNotFoundError:
    # package is not installed
    pass

from .monitor import *
from .dandi_example_data import *
from .array_info import *
from .cap_memory import *
