import numpy as np
import torch
from sklearn.metrics import mean_squared_error

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# evaluation classifier
def evaluation_gender(data, label, model):
    model.eval()
    pred = model(data)
    pred_out = pred.argmax(1)
    acc = round(sum(pred_out == label).item()/(pred_out.shape[0]) * 100, 2)
    pred_s0_s1_ratio = ((sum(pred_out == 0).item() + 1e-2)/(sum(pred_out == 1).item() + 1e-2))
    return acc, pred_s0_s1_ratio

def validate_fairness(model, df_val, df_sensitive_attr, s1_known, s0_known, device):
    model.eval()
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"])).to(device)
        test_item_total = torch.tensor(np.array(df_val["item_id"])).to(device)
        pred_total = model(test_user_total, test_item_total)
        pred_total = pred_total.cpu().detach()
        uniq_count= 0
        fairness_count = 0
        naive_fairness_dict = {i:[] for i in df_sensitive_attr["gender"].unique()}
        df_val_with_index = df_val.reset_index()
        group_val = df_val_with_index.groupby("user_id")
        df_sensitive_dict = df_sensitive_attr.set_index("user_id")
        y_true_all = []
        y_pred_all = []
        for name, group in group_val:
            test_rating = np.array(group["label"]).astype(int)
            y_hat = pred_total[group.index]
            y_true_all += list(test_rating)
            y_pred_all += y_hat.tolist()
            if len(np.unique(test_rating)) != 1:
                y_hat_sort_id = y_hat.sort(descending=True).indices
                label_rank = test_rating[y_hat_sort_id]
                uniq_count += 1
            if (name in s0_known) or (name in s1_known  ): 
                gender = int(df_sensitive_dict.iloc[name]["gender"])
                naive_fairness_dict[gender] += y_hat.tolist()
                fairness_count += 1

        naive_gender_unfairness = float(np.abs(np.mean(naive_fairness_dict[1]) - (np.mean(naive_fairness_dict[0]))))
        rmse_result = RMSE(y_true_all, y_pred_all)
    return rmse_result, naive_gender_unfairness

def test_fairness(model, df_val, df_sensitive_attr, device):
    model.eval()
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"])).to(device)
        test_item_total = torch.tensor(np.array(df_val["item_id"])).to(device)
        pred_total = model(test_user_total, test_item_total)
        pred_total = pred_total.cpu().detach()
        uniq_count= 0
        naive_fairness_dict = {i:[] for i in df_sensitive_attr["gender"].unique()}
        df_val_with_index = df_val.reset_index()
        group_val = df_val_with_index.groupby("user_id")
        df_sensitive_dict = df_sensitive_attr.set_index("user_id")
        y_true_all = []
        y_pred_all = []
        for name, group in group_val:
            test_rating = np.array(group["label"]).astype(int)
            y_hat = pred_total[group.index] 
            y_true_all += list(test_rating)
            y_pred_all += y_hat.tolist()
            if len(np.unique(test_rating)) != 1:
                y_hat_sort_id = y_hat.sort(descending=True).indices
                label_rank = test_rating[y_hat_sort_id]
                uniq_count += 1
            gender = int(df_sensitive_dict.iloc[name]["gender"])
            naive_fairness_dict[gender] += y_hat.tolist()

        naive_gender_unfairness = float(np.abs(np.mean(naive_fairness_dict[1]) - (np.mean(naive_fairness_dict[0]))))
        rmse_result = RMSE(y_true_all, y_pred_all)
        
    return rmse_result, naive_gender_unfairness

