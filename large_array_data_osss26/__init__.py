from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("large_array_data_osss26")
except PackageNotFoundError:
    # package is not installed
    pass
