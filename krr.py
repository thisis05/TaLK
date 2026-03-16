import torch
import torch.nn as nn



class KRRwithLM_Outer(nn.Module):
    def __init__(self, kernel, ridge, model):
        super(KRRwithLM_Outer, self).__init__()
        self.kernel   = kernel
        self.ridge    = ridge
        self.lm = model
    
    def forward(self, G_t, G_s, y_t, y_s, E_t, E_s):
        G_t = G_t.detach()
        G_s = self.lm(inputs_embeds = G_s)
        G_s = G_s.last_hidden_state[:, 0, :]
        K_ss      = self.kernel(G_s, G_s, E_s, E_s)
        K_ts      = self.kernel(G_t, G_s, E_t, E_s)
        n        = torch.tensor(len(G_s), device = G_s.device)
        regulizer = self.ridge * torch.trace(K_ss) * torch.eye(n, device=G_s.device) / n
        b         = torch.linalg.solve(K_ss + regulizer, y_s)
        pred      = torch.matmul(K_ts, b)
        
        return pred
