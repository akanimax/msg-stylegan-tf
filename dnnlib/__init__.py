from . import tflib  # added manually to correct the original mistake
from . import submission
from .submission.run_context import RunContext
from .submission.submit import (PathType, SubmitConfig, SubmitTarget,
                                get_path_from_template, submit_run)
from .util import EasyDict

submit_config: SubmitConfig = None  # Package level variable for SubmitConfig which is only valid when inside the run function.
