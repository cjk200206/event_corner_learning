import models_aegnn.layer
import models_aegnn.networks
from models_aegnn.recognition import RecognitionModel

################################################################################################
# Access functions #############################################################################
################################################################################################
import pytorch_lightning as pl


def by_task(task: str) -> pl.LightningModule.__class__:
    if task == "recognition":
        return RecognitionModel
    else:
        raise NotImplementedError(f"Task {task} is not implemented!")
