import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import mpr
import numpy as np
import pandas as pd

def plot_embeddings(model, sensitive_df, device, title="User Embeddings"):
    model.eval()
    with torch.no_grad():
        # get user embeddings
        user_ids = torch.arange(model.user_emb.num_embeddings).to(device)
        embs = model.user_emb(user_ids).cpu().numpy()
    
    # map gender labels to embeddings
    gender_map = sensitive_df.set_index('user_id')['gender'].to_dict()
    colors = [gender_map.get(i, 2) for i in range(len(embs))] # 2 for unknown
    
    # Run t-SNE 
    tsne = TSNE(n_components=2, random_state=42)
    embs_2d = tsne.fit_transform(embs)
    
    # Plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=colors, cmap='coolwarm', alpha=0.6, s=1)
    plt.colorbar(scatter, ticks=[0, 1], label='Gender (0: Female, 1: Male)')
    plt.title(title)
    plt.show()

best_model = mpr.matrixFactorization(
    mpr.num_uniqueUsers, 
    mpr.num_uniqueLikes, 
    mpr.emb_size
).to(mpr.device)

# load weights from file
state_dict = torch.load('debug_MPR_thresh_eval/best_model', map_location=mpr.device)

# load weights into model
best_model.load_state_dict(state_dict)

orig_sensitive_attr = pd.read_csv(mpr.sensitive_csv_path,dtype=np.int64)

plot_embeddings(best_model, orig_sensitive_attr, mpr.device, title="Fair MF-MPR Embeddings")