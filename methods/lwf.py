from models import LoRAAdapter, CLIPWrapper
from methods.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from copy import deepcopy
from data import TaskData, TaskDataLoader
from tqdm import tqdm
class LwF_LoRA(BaseTrainer):
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__(wrapper, config)
        self.wrapper = wrapper
        self.add_lora()

        #? tách riêng các thành phần của model
        self.vision_model = wrapper.model.vision_model
        self.org_visual_proj = wrapper.model.visual_projection
        self.text_model = wrapper.model.text_model
        self.org_text_proj = wrapper.model.text_projection

        #? lưu head theo task
        self.task_heads = nn.ModuleDict()
        
        #? setpoint cho old tasks:
        #@ {old_task_id: tensor [N, D]}
        self.old_task_setpoints = {}
        
        #? nhiệt độ distillation
        self.distill_temp = self.config.train.distill_temp

        #? trọng số loss cũ
        self.lambda_old = getattr(self.config.train, "lambda_old", 1.0)
        
        #? id của các task đã train
        self.trained_task_id = []
    
    def add_lora(self):
        device = self.wrapper.model.device

        # ? Freeze model lại
        for param in self.wrapper.model.parameters():
            param.requires_grad = False

        #? thêm LoRA vào các layer q_proj, v_proj
        for i in range(self.config.model.num_layers):
            for layer_type in ['q_proj', 'v_proj']:
                #* Vision
                attn = self.wrapper.model.vision_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRAAdapter(original, r= self.config.train.r))

                # #*Text
                attn = self.wrapper.model.text_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRAAdapter(original, r=self.config.train.r))
        self.wrapper.model.to(device)
        
    def _build_new_head_if_needed(self, task_id):
        device = self.wrapper.model.devcie
        #? neếu chưa từng train trên task này thì sẽ tạo head mới
        if task_id not in self.trained_tasks_ids:
            #? với từng loại head (text, visual)
            new_visual_head = deepcopy(self.org_visual_proj)
            new_visual_head.load_state_dict(self.org_visual_proj.state_dict())
            new_visual_head = new_visual_head.to(device)

            new_text_head = deepcopy(self.org_text_proj)
            new_text_head.load_state_dict(self.org_text_proj.state_dict())
            new_text_head = new_text_head.to(device)

            #? chỉ train head mới này
            #! cần check lại vì mình nhớ là train hết mọi head
            for p in new_visual_head.parameters():
                p.requires_grad = True
            for p in new_text_head.parameters():
                p.requires_grad = True

            self.task_heads[(str(task_id), 'visual')] = new_visual_head
            self.task_heads[(str(task_id), "text")] = new_text_head
    
    def _init_old_tasks_setpoints(self, images, labels):
        pass

    def train(self, task, task_id = None):
        optimizer = self.optimizer(self.wrapper.model.parameters(),
                                   lr = float(self.config.train.lr),
                                   weight_decay = float(self.config.train.weight_decay))

        criterion = nn.CrossEntropyLoss()
        prompts = [f"a photo of a {name}" for name in task['label_names']]

        #* =======================Data and Dataloader============================
        train_data = TaskData(task, "train", image_processor= self.wrapper.processor.image_processor)
        test_data = TaskData(task, "test", image_processor= self.wrapper.processor.image_processor)
        train_loader = TaskDataLoader(
            train_data,
            batch_size= self.config.datasets.batch_size,
            num_workers=self.config.datasets.num_workers,
            pin_memory= True)
        test_loader = TaskDataLoader(
            test_data,
            batch_size = self.config.datasets.batch_size,
            num_workers=self.config.datasets.num_workers,
            pin_memory=True
        )
        device = self.wrapper.model.device

        #* =====================TRAIN=======================================
        self.wrapper.model.train()
        train_loss = 0
        #@ cần phải deepcopy lại LoRA trước khi train, để có thể init được setpoint
        old_LoRA = self.wrapper.split_and_get_lora()
        old_LoRA_copy = deepcopy(old_LoRA)
        self.wrapper.load_lora(old_LoRA)
        #@ create new head
        if task_id not in self.trained_task_id:
            self._build_new_head_if_needed(task_id)
            new_visual_head = self.task_heads[( str(task_id), 'visual' )]
            new_text_head = self.task_heads[(str(task_id), 'text')]
        for epoch in range(self.config.train.max_epoch):
            for image_tensors, labels in tqdm(train_loader,  desc=f"Train Epoch {epoch+1}", leave=False):
                labels = labels.to(device)
                optimizer.zero_grad()
                #@ calculate loss_term_1
                #? split LoRA
                current_LoRA = self.wrapper.split_and_get_lora()
                #? load old LoRA
                self.wrapper.load_lora(old_LoRA_copy)
                
                #? save the setpoints of the previous heads
                prev_heads_setpoints = {}
                with torch.inference_mode():
                    #? cho 2 backbone chạy qua data đã
                    text_model_out = self.wrapper.forward_text_text_model(prompts)
                    visual_model_out = self.wrapper.forward_image_vision_model(image_tensors)
                    for trained_task_id in range(task_id):
                        visual_out = self.task_heads[(str(trained_task_id), 'visual')](visual_model_out)
                        text_out = self.task_heads[(str(trained_task_id), 'text')](text_model_out)

                        prev_heads_setpoints[(str(trained_task_id), 'visual')] = visual_out
                        prev_heads_setpoints[(str(trained_task_id), 'text')] = text_out

                self.wrapper.load_lora(current_LoRA)

                loss_term_1 = torch.tensor(0)
                for 

                #@ calculate loss_term_2
                text_features = self.wrapper.encode_text(prompts)
                logits = self._pred(image_tensors, text_features)
                loss_term_2 = criterion(logits, labels)
                
                