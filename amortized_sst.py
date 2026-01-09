import torch
from torch import nn
import numpy as np
from tqdm import tqdm
class AmortizedSST(nn.Module):
    """
    Amortized model that takes user embeddings and prior belief about sensitive attribute
    and predicts the sensitive attribute using a feedforward neural network.

    Args:
        emb_size (int): Size of the user embedding vector.
    Inputs:
        z_u (torch.Tensor): User embedding of shape (batch_size, emb_size).
        p0 (float): Prior belief about the sensitive attribute (e.g., probability of being in class 1).
    Outputs:
        torch.Tensor: Predicted sensitive attribute probabilities of shape (batch_size, 1).
    """
    def __init__(self, emb_size):
        super(AmortizedSST, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(emb_size + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z_u, p0):
        # Concatenate embedding and prior belief
        p0_expanded = torch.full((z_u.size(0), 1), p0).to(z_u.device) # expand to match batch size
        x = torch.cat([z_u, p0_expanded], dim=1)
        return self.fc(x)
    
def train_amortized_sst(sst_model, mf_model, c0, c1, epochs=20, device='cuda'):
    optimizer = torch.optim.Adam(sst_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss(reduction='none') # none to apply weights later
    
    # combined known sensitive attribute data
    user_ids = torch.cat([torch.tensor(c0), torch.tensor(c1)]).to(device)
    labels = torch.cat([torch.zeros(len(c0)), torch.ones(len(c1))]).to(device)
    
    # empirical ratio
    hat_p1 = len(c1) / (len(c0) + len(c1))
    
    sst_model.train()
    for _ in tqdm(range(epochs), desc="Training Amortized SST"):
        # get embeddings
        z_u = mf_model.user_embeddings(user_ids).detach()
        
        # sample random prior for each batch
        p1 = np.random.uniform(0.1, 0.9)
        
        # fwd pass
        preds = sst_model(z_u, p1).view(-1)
        
        # weighting for importance sampling
        # Weight for class 1: p1 / hat_p1
        # Weight for class 0: (1-p1) / (1-hat_p1)
        weights = torch.where(labels == 1, p1 / hat_p1, (1 - p1) / (1 - hat_p1))
        
        loss = (criterion(preds, labels) * weights).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()