import numpy as np
import random
from model import PretrainModel
import torch
from sklearn.metrics import accuracy_score
import copy
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from transformers import AutoTokenizer
import torch.nn as nn

class PretrainModelManager:  

    def __init__(self, args, data):
        self.set_seed(args.seed)
        self.args = args
        self.data = data
        self.model = PretrainModel(args, data)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.best_eval_score = 0
        self.optimizer = self.get_optimizer(args)
        
        self.optimization_steps = int(len(data.train_examples) / args.pretrain_batch_size) * args.num_pretrain_epochs
        self.num_warmup_steps = int(args.warmup_proportion * self.optimization_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.num_warmup_steps,
                                                    num_training_steps=self.optimization_steps)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_pre)
        return optimizer

    def save_model(self, save_path):
        self.model.save_backbone(save_path)

    def train(self):
        wait = 0
        best_model = None

        for epoch in range(self.args.num_pretrain_epochs):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for _, batch in enumerate(self.data.pretrain_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_coarse, _ = batch
                X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}

                with torch.set_grad_enabled(True):
                    logits = self.model(X)["logits"]
                    loss_src = self.model.loss_ce(logits, label_coarse)
                    lossTOT = loss_src
                    lossTOT.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    tr_loss += lossTOT.item()
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.eval()
            print('score', eval_score)
            
            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= self.args.wait_patient:
                    break
                
        self.model = best_model
        if self.args.save_premodel:
            self.save_model(self.args.save_premodel_path)

    def eval(self):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.data.n_coarse)).to(self.device)

        for batch in self.data.eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_coarse, _ = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.set_grad_enabled(False):
                logits = self.model(X)['logits']
                total_labels = torch.cat((total_labels, label_coarse))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc