import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from evaluation import validate_fairness, test_fairness

from tqdm import tqdm

def train_fair_mf_mpr(
    model,
    sst_model,
    df_train,
    epochs,
    lr,
    weight_decay,
    batch_size,
    beta,
    valid_data,
    test_data,
    resample_range,
    oracle_sensitive_attr,
    top_K,
    fair_reg,
    c0_known,
    c1_known,
    device,
    evaluation_epoch=3,
    unsqueeze=False,
    shuffle=True,
    rmse_thresh=None
):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    test_rmse_in_that_epoch = float("inf")
    best_val_unfairness = float("inf")
    naive_unfairness_test_in_that_epoch = float("inf")
    val_rmse_in_that_epoch = float("inf")
    best_epoch = 0
    achieve_rmse_thresh = False

    num_batches = len(df_train) // batch_size

    for epoch in tqdm(range(epochs), desc="Training Fair MF with MPR"):
        loss_total = 0.0
        fair_reg_total = 0.0

        indices = np.arange(len(df_train))
        if shuffle:
            np.random.shuffle(indices)

        for batch_idx in range(num_batches):
            batch_ids = indices[batch_idx*batch_size : (batch_idx+1)*batch_size]
            data_batch = df_train.iloc[batch_ids].reset_index(drop=True)

            # Prepare inputs
            train_ratings = torch.FloatTensor(data_batch["label"].values).to(device)
            train_user_input = torch.LongTensor(data_batch["user_id"].values).to(device)
            train_item_input = torch.LongTensor(data_batch["item_id"].values).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)

            # Forward
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat.view(-1), train_ratings.view(-1))

            current_user_embs = model.user_embeddings(train_user_input)

            # Fairness regularization using log-sum-exp trick
            C = torch.tensor(0.0, device=device)
            reg_sum = torch.tensor(0.0, device=device)

            fair_diffs = []
            eps = 1e-8

            for p0 in resample_range:
                # Use probability for weighted means
                s_hat = sst_model(current_user_embs, float(p0))

                # Mean G1: weighted by probability s_hat
                # Mean G0: weighted by probability (1 - s_hat)
                mu_1 = torch.sum(y_hat * s_hat) / (torch.sum(s_hat) + eps)
                mu_0 = torch.sum(y_hat * (1 - s_hat)) / (torch.sum(1 - s_hat) + eps)

                fair_diffs.append(torch.abs(mu_1 - mu_0) / beta)

            fair_diffs_tensor = torch.stack(fair_diffs)
            C = torch.max(fair_diffs_tensor).detach() # C acts as constant

            reg_sum = torch.sum(torch.exp(fair_diffs_tensor - C))
            fair_regulation = fair_reg * beta * (torch.log(reg_sum + eps) + C)             

            # Backprop
            total_loss = loss + fair_regulation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_total += loss.item()
            fair_reg_total += fair_regulation.item()

        print(f"Epoch {epoch}: avg loss {loss_total/num_batches:.4f}, avg fair reg {fair_reg_total/num_batches:.4f}")

        # Evaluation
        if epoch % evaluation_epoch == 0:
            rmse_val, naive_unfairness_val = validate_fairness(
                model, valid_data, oracle_sensitive_attr, c0_known, c1_known, device
            )
            rmse_test, naive_unfairness_test = test_fairness(
                model, test_data, oracle_sensitive_attr, device
            )
            
            print(f"Validation RMSE: {rmse_val:.4f}, Partial Valid Unfairness: {naive_unfairness_val:.4f}")
            print(f"Test RMSE: {rmse_test:.4f}, Unfairness: {naive_unfairness_test:.4f}")

            if rmse_thresh is not None and rmse_val < rmse_thresh:
                achieve_rmse_thresh = True
                if naive_unfairness_val < best_val_unfairness:
                    best_epoch = epoch
                    best_val_unfairness = naive_unfairness_val
                    val_rmse_in_that_epoch = rmse_val
                    test_rmse_in_that_epoch = rmse_test
                    naive_unfairness_test_in_that_epoch = naive_unfairness_test
                    best_model = copy.deepcopy(model)

    if not achieve_rmse_thresh:
        best_epoch = -1
        best_model = copy.deepcopy(model)

    return (
        val_rmse_in_that_epoch,
        test_rmse_in_that_epoch,
        best_val_unfairness,
        naive_unfairness_test_in_that_epoch,
        best_epoch,
        best_model
    )
