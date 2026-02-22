import torch 
import torch.nn as nn 
from einops import rearrange


class Prototype_Prompt_Encoder(nn.Module):
    def __init__(self, feat_dim=256, 
                        hidden_dim_dense=128, 
                        hidden_dim_sparse=128, 
                        size=64, 
                        num_tokens=8):
                
        super(Prototype_Prompt_Encoder, self).__init__()
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)
        
        self.relu = nn.ReLU()

        self.sparse_fc_1 = nn.Conv1d(size*size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_tokens, 1)
        
        
        pn_cls_embeddings = [nn.Embedding(num_tokens, feat_dim) for _ in range(2)] # one for positive and one for negative 

            
        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)
                
    def forward(self, feat, prototypes, cls_ids):
  
        cls_prompts = prototypes.unsqueeze(-1)
        cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)

        
        feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1)

        # compute similarity matrix 
        sim = torch.matmul(feat, cls_prompts)
        
        # compute class-activated feature
        feat =  feat + feat*sim

        feat_sparse = feat.clone()
        
        # compute dense embeddings
        one_hot = torch.nn.functional.one_hot(cls_ids-1,7) 
        # ---- FIX: make one_hot match number of prototypes (classes) ----
        # cls_ids expected to be 1..C in this project, so convert to 0-index
        C = prototypes.size(0)               # dynamic number of classes
        cls_ids = cls_ids.long().view(-1)    # (B,)
        idx = (cls_ids - 1).clamp(0, C - 1)  # (B,) in [0..C-1]

        one_hot = torch.zeros((idx.size(0), C), device=idx.device, dtype=torch.float32)
        one_hot.scatter_(1, idx.unsqueeze(1), 1.0)
        # ---------------------------------------------------------------
        if feat.dim() == 4 and feat.size(1) == 1:
            feat = feat.squeeze(1)

        feat = rearrange(feat, 'b (h w) c -> b c h w', h=64, w=64)
        dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(feat)))
        
        # compute sparse embeddings
        feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c')
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        # num_cls must match how many class prototypes we have (C)
        C = prototypes.size(0)  # dynamic classes (WetCat=1, EndoVis=7)

        # If sparse_embeddings is (B*C, N, Cdim) -> reshape to (B, C, N, Cdim)
        if sparse_embeddings.dim() == 3:
            sparse_embeddings = rearrange(
                sparse_embeddings,
                '(b num_cls) n c -> b num_cls n c',
                num_cls=C
            )
        
        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)
        

        sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()
            
        sparse_embeddings = rearrange(sparse_embeddings,'b num_cls n c -> b (num_cls n) c')
        
        return dense_embeddings, sparse_embeddings
    



class Learnable_Prototypes(nn.Module):
    def __init__(self, num_classes=7 , feat_dim=256):
        super(Learnable_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)
        
    def forward(self):
        return self.class_embeddings.weight