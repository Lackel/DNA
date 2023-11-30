import numpy as np
import torch

class MemoryBank(object):
    
    def __init__(self, n, dim, m):
        self.n = n
        self.dim = dim 
        self.m = m
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.coarse_labels = torch.LongTensor(self.n)
        self.ptr = 0

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True, rank=False):
        # Mine the top-k nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        ind = indices.copy()
        sort_feature = torch.argsort(torch.tensor(features).cuda(), dim=1, descending=True)
        sort_feature, _ = torch.sort(sort_feature[:, :self.m], dim=1)
        # Three Constraints
        for i in range(indices.shape[0]):
            arg_feature = sort_feature[i, :]
            for j in range(indices.shape[1]):
                # Rank Statistic Constraint
                if rank:
                    arg_neighbour = sort_feature[indices[i][j], :]
                    rank_diff = arg_feature - arg_neighbour
                    flag = torch.sum(torch.abs(rank_diff))
                    if flag > 0:
                        ind[i][j] = -1
                # Label Constraint
                if self.coarse_labels[i] != self.coarse_labels[indices[i][j]]:
                    ind[i][j] = -1
                # Reciprocal Constraint
                if i not in indices[indices[i][j]][:]:
                    ind[i][j] = -1

        # Evaluate retrieval accuracy
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for evaluation
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            count = 0
            term = 0
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]-1):
                    if ind[i][j+1] != -1:
                        count += 1
                        if neighbor_targets[i][j] == anchor_targets[i][j]:
                            term += 1
            accuracy = term / count
            return ind, accuracy
        
        else:
            return ind

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets, coarse_labels):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.coarse_labels[self.ptr:self.ptr+b].copy_(coarse_labels.detach())
        self.ptr += b

    def up(self, feature, index, label, coarse_labels):
        for index, item in enumerate(index):
            self.features[item].copy_(feature[index].detach())
            self.targets[item].copy_(label[index].detach())
            self.coarse_labels[item].copy_(coarse_labels[index].detach()) 


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        batch = tuple(t.cuda(non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, coarse_labels, label_ids = batch
        X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
        feature = model(X)
        memory_bank.update(feature, label_ids, coarse_labels)