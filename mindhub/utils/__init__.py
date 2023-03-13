from . import download, path

from .download import DownLoad, set_default_download_root, get_default_download_root
from .path import detect_file_type

__all__ = []
__all__.extend(download.__all__)
__all__.extend(path.__all__)
