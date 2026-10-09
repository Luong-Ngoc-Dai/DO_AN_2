"""CAE project; use writable plotting caches on read-only cloud home directories."""
import os
from pathlib import Path
import tempfile

# Set before importing TensorFlow/Keras, which may import Matplotlib internally.
_cache = Path(tempfile.gettempdir())/'do_an_2_cache'
for _name, _subdir in [('MPLCONFIGDIR','matplotlib'), ('XDG_CACHE_HOME','xdg')]:
    if _name not in os.environ:
        _path = _cache/_subdir
        _path.mkdir(parents=True,exist_ok=True)
        os.environ[_name]=str(_path)
