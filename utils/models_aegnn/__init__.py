from . import networks,layer
from .recognition import RecognitionModel

################################################################################################
# Access functions #############################################################################
################################################################################################
import pytorch_lightning as pl


def by_task(task: str) -> pl.LightningModule.__class__:
    if task == "recognition":
        return RecognitionModel
    else:
        raise NotImplementedError(f"Task {task} is not implemented!")
