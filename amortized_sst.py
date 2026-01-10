import torch
from torch import nn
from tqdm import tqdm
import random

from torch.utils.data import DataLoader, TensorDataset
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
        super().__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(emb_size, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z_u, p0):
            if not isinstance(p0, torch.Tensor):
                p0 = torch.full((z_u.size(0), 1), p0).to(z_u.device)
            elif p0.dim() == 1:
                p0 = p0.view(-1, 1)

            # Force the model to process embeddings in a prior-blind way first
            features = self.embedding_net(z_u)
            combined = torch.cat([features, p0], dim=1)

            return self.classifier(combined)

def alpha_schedule(epoch, alpha_max=0.5, warmup_epochs=10):
    return min(alpha_max, alpha_max * epoch / warmup_epochs)

def fair_reg_schedule(epoch, fair_reg_max=200.0, warmup=10):
    return fair_reg_max * min(1.0, epoch / warmup)

def train_amortized_sst(
    sst_model,
    mf_model,
    s0,
    s1,
    epochs=30,
    batch_size=1024,
    alpha_max=0.1,  # weight for prior matching constraint
    device="cuda"
):
    optimizer = torch.optim.Adam(sst_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss(reduction="none")

    # data
    user_ids = torch.cat([torch.tensor(s0), torch.tensor(s1)]).long()
    labels = torch.cat([torch.zeros(len(s0)), torch.ones(len(s1))]).float()
    hat_p1 = len(s1) / (len(s0) + len(s1))

    dataset = TensorDataset(user_ids, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # fixed priors for stable gradient estimation
    anchor_priors = [
        (0.02, 2.5),
        (0.05, 2.0),
        (0.1, 1.5),
        (0.3, 1.0),
        (0.5, 1.0),
        (0.7, 1.0),
        (0.9, 1.5),
        (0.95, 2.0),
        (0.98, 2.5),
    ]
    sst_model.train()
    for epoch in range(epochs):
        alpha = alpha_schedule(epoch, alpha_max=alpha_max, warmup_epochs=10)    
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} | Alpha: {alpha:.4f}")
        
        for batch_u, batch_l in pbar:
            batch_u, batch_l = batch_u.to(device), batch_l.to(device)
            
            # don't update mf weights
            z_u = mf_model.user_emb(batch_u).detach()
            
            batch_total_loss = 0
            
            # train model on multiple worlds
            for p_val, w in anchor_priors:
                # forward pass
                preds = sst_model(z_u, p_val).view(-1)
                
                # importance weighting for specific prior
                weights = torch.where(
                    batch_l == 1,
                    p_val / hat_p1,
                    (1 - p_val) / (1 - hat_p1)
                ).clamp(0.1, 10.0)
                
                # data loss
                data_loss = (criterion(preds, batch_l) * weights).mean()
                
                # force the model to respect prior
                prior_match = (preds.mean() - p_val).pow(2) * (1 + abs(p_val - 0.5))

                batch_total_loss += w * (data_loss + alpha * prior_match)

            # backward pass for all priors combined
            optimizer.zero_grad()
            batch_total_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_total_loss.item()
            pbar.set_postfix({"loss": f"{batch_total_loss.item():.4f}"})

def evaluate_amortized_sst(sst_model, mf_model, s0_test, s1_test, device):
    sst_model.eval()
    # Test on a neutral prior (0.5) and extreme priors
    test_priors = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    user_ids = torch.cat([torch.tensor(s0_test), torch.tensor(s1_test)]).to(device)

    with torch.no_grad():
        z_u = mf_model.user_emb(user_ids)
        
        print("\n[Calibration] Amortized SST Predictions under Different Priors:")
        for p in test_priors:
            preds = sst_model(z_u, p).view(-1)
            print(
                f"Prior {p:.2f} | "
                f"mean(pred)={preds.mean().item():.3f}, "
                f"std(pred)={preds.std().item():.3f}"
            )

def refine_sst(sst_model, mf_model, s0_known, s1_known, resample_range, fair_diffs, device, lr=1e-4):
    """
    Adversarially refines the SST on the current 'worst' worlds.
    """
    sst_model.train()
    optimizer = torch.optim.Adam(sst_model.parameters(), lr=lr) # Lower LR for fine-tuning
    criterion = nn.BCELoss(reduction="none")
    
    # idnetify worst priors
    _, top_indices = torch.topk(fair_diffs.detach(), k=3)
    worst_priors = resample_range[top_indices].tolist()
    
    # avoid forgetting by mixing with stable priors
    active_priors = list(set(worst_priors + [0.1, 0.5, 0.9]))
    
    # data
    user_ids = torch.cat([torch.tensor(s0_known), torch.tensor(s1_known)]).to(device)
    labels = torch.cat([torch.zeros(len(s0_known)), torch.ones(len(s1_known))]).to(device)
    hat_p1 = len(s1_known) / (len(s0_known) + len(s1_known))

    # single pass over known data
    z_u = mf_model.user_emb(user_ids).detach()
    
    for p_val in tqdm(active_priors, desc="[Adversarial Update] Refining SST"):
        for _ in range(3):
            preds = sst_model(z_u, p_val).view(-1)
            weights = torch.where(labels == 1, p_val / hat_p1, (1 - p_val) / (1 - hat_p1))
            
            # high alpha to respect prior strongly during refinement
            loss = (criterion(preds, labels) * weights).mean() + 1.0 * (preds.mean() - p_val).pow(2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    sst_model.eval()