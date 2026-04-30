from ..models import CLIPWrapper
from ..config import Config

class ContinualLearningMethod:
    def __init__(self):
        super().__init__()
        self.criterion = None
        self.wrapper = None
        self.requires_task_id = False
        self.config = None
    
    def set_config(self, config:Config):
        self.config = config
    
    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_wrapper(self, wrapper):
        self.wrapper: CLIPWrapper = wrapper
    
    #@ đặc biệt dùng khi eval, vì không cần encode text_tokenized nhiều lần
    def compute_loss_inference_mode(self, images, labels, text_features):
        logits = self.wrapper.forward_with_text_feature(text_features, images)
        loss_ce = self.criterion(logits, labels)
        return loss_ce

    def compute_loss(self, images, labels, text_tokenized):
        raise NotImplementedError
    
    def initialize(self, task_id):
        if self.requires_task_id:
            raise NotImplementedError