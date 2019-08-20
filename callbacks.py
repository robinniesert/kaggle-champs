import torch
from fastai.basic_train import Learner, LearnerCallback

class GradientClipping(LearnerCallback):
    "Gradient clipping during training."
    def __init__(self, learn:Learner, clip:float = 0., 
                 start_it:int = 100):
        super().__init__(learn)
        self.clip, self.start_it = clip, start_it

    def on_backward_end(self, iteration, **kwargs):
        "Clip the gradient before the optimizer step."
        if self.clip and (iteration > self.start_it): 
            torch.nn.utils.clip_grad_norm_(
                self.learn.model.parameters(), self.clip)