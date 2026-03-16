import torch
import math
import torch.nn as nn


class StructureBasedNeuralTangentKernel(nn.Module):
    def __init__(self,  K=2, L=2, scale='add' ):
        super(StructureBasedNeuralTangentKernel, self).__init__()
        self.K = K
        self.L = L
        self.scale  = scale

    def _degree(self, E):
        if E.is_sparse:
            deg = torch.sparse.sum(E, dim=1).to_dense()
        else:
            deg = E.sum(dim=1)
        return deg

    def aggr(self, S, E1, E2, scale_mat):

        if E1.is_sparse:
            S = torch.sparse.mm(E1, S)       # sparse @ dense
        else:
            S = E1 @ S                       # dense @ dense

        if E2.is_sparse:
            S = torch.sparse.mm(
                E2.transpose(0, 1),
                S.transpose(0, 1)
            ).transpose(0, 1)
        else:
            S = S @ E2.transpose(0, 1)       # dense @ dense   

        return S * scale_mat


    def update_sigma(self, S, diag1, diag2):
        S = S / diag1[:, None] / diag2[None, :]
        S = torch.clip(S, -0.9999, 0.9999)
        S_relu = (S * (math.pi - torch.arccos(S)) +
                  torch.sqrt(1 - S * S)) / math.pi
        degree_sigma = (math.pi - torch.arccos(S)) / math.pi
        S_new = S_relu * diag1[:, None] * diag2[None, :]
        return S_new, degree_sigma
    
    def update_diag(self, S):
        diag = torch.sqrt(torch.diag(S))
        S = S / diag[:, None] / diag[None, :]
        S = torch.clip(S, -0.9999, 0.9999)
        S_relu = (S * (math.pi - torch.arccos(S)) +
                  torch.sqrt(1 - S * S)) / math.pi
        S_new = S_relu * diag[:, None] * diag[None, :]
        return S_new, diag
    
    def diag_self(self, g, E, eps = 1e-8):
        n = g.shape[0]
        
        if self.scale == 'add':
            inv_deg = torch.ones(n, 1, device=g.device)
        else:
            if E.is_sparse:
                deg = torch.sparse.sum(E, dim=1).to_dense().unsqueeze(1)
            else:
                deg = E.sum(dim=1).unsqueeze(1)
            
            inv_deg = 1.0 / deg.clamp(min=1.0) # (n, 1)

        diag_list = []
        
        H = g 

        for k in range(self.K):
            if E.is_sparse:
                H = torch.sparse.mm(E, H)
            else:
                H = torch.matmul(E, H)
            
            H = H * inv_deg
            current_diag = torch.sum(H * H, dim=1) 
            current_diag_sqrt = torch.sqrt(current_diag.clamp_min(eps))
            
            for l in range(self.L):
                diag_list.append(current_diag_sqrt)

        return diag_list
        

    def diag(self, g, E):
        """
        g: (n, d)
        E: (n, n) sparse adjacency 
        """
        n = E.shape[0]

        if self.scale == 'add':
            scale_mat = 1.0
        else:
            deg = self._degree(E).clamp_min(1.0)
            scale_mat = 1.0 / (deg[:, None] * deg[None, :])  # (n, n)

        diag_list = []
        sigma = torch.matmul(g, g.t())  # (n, n)

        for k in range(self.K):
            sigma = self.aggr(sigma, E, E, scale_mat)  
            for l in range(self.L):
                sigma, diag = self.update_diag(sigma)
                diag_list.append(diag)
        return diag_list


    def nodes_gram(self, g1, g2, E1, E2):
        """
        g1: (n1, d)
        g2: (n2, d)
        E1: (n1, n1) sparse
        E2: (n2, n2) sparse
        """
        n1, n2 = g1.size(0), g2.size(0)

        if self.scale == 'add':
            scale_mat = 1.0
        else:
            deg1 = self._degree(E1).clamp_min(1.0)  
            deg2 = self._degree(E2).clamp_min(1.0)  
            scale_mat = 1.0 / (deg1[:, None] * deg2[None, :])  # (n1, n2)

        sigma = torch.matmul(g1, g2.t())  # (n1, n2)
        theta = sigma.clone()

        diag_list1 = self.diag_self(g1, E1)
        diag_list2 = self.diag_self(g2, E2)

        for k in range(self.K):
            sigma = self.aggr(sigma, E1, E2, scale_mat)
            theta = self.aggr(theta, E1, E2, scale_mat)

            for l in range(self.L):
                idx = k * self.L + l
                sigma, degree_sigma = self.update_sigma(
                    sigma,
                    diag_list1[idx],
                    diag_list2[idx]
                )
                theta = theta * degree_sigma + sigma

        return theta