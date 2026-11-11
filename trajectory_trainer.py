"""Memory savings for complete screenshot/action trajectories on one GPU."""

from torch.nn import functional as F
from trl import SFTTrainer

from qwen_model import project_selected_tokens


class TrajectoryTrainer(SFTTrainer):
    """Compute vocabulary logits only where an action token is supervised."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Our loss is a microbatch mean. Trainer must divide it by accumulation
        # steps, rather than assuming the model normalizes across microbatches.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        # State at position k-1 predicts label k. The union also supports padding
        # when different trajectories in a batch have different action positions.
        positions = (labels[:, 1:] != -100).any(dim=0).nonzero(as_tuple=True)[0]
        targets = labels[:, positions + 1]

        with project_selected_tokens(model, positions):
            outputs = model(**inputs)

        loss = F.cross_entropy(
            outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss
