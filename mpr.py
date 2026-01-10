import argparse
import numpy as np
import pandas as pd
import os
import random
import csv
import torch
from fairness_training import train_fair_mf_mpr
from collaborative_models import matrixFactorization
from amortized_sst import AmortizedSST, train_amortized_sst, evaluate_amortized_sst

# RETRAIN SST ON KNOWN WORST PRIOR EVERY 50 EPOCHS

parser = argparse.ArgumentParser(description='fairRec')
parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='7',
                        help="device id to run")
parser.add_argument("--beta", 
                    type = float,
                    default = 0.005,
                    help = "Beta in KL-Closed form solution.")
parser.add_argument("--embed_size", type=int, default= 64, help= "the embedding size of MF")
parser.add_argument("--output_size", type=int, default= 1, help="the output size of MF")
parser.add_argument("--num_epochs", type=int, default= 200, help= "the max epoch of training")
parser.add_argument("--learning_rate", type= float, default= 1e-3, help="the learning rate for MF model")
parser.add_argument("--batch_size", type= int, default= 32768, help= "the batchsize for training")
parser.add_argument("--evaluation_epoch", type= int, default= 3, help= "the evaluation epoch")
parser.add_argument("--weight_decay", type= float, default= 1e-7, help= "the weight_decay for training")
parser.add_argument('--seed', type=int, default=1, help="the random seed")
parser.add_argument("--saving_path", type=str, default= "./debug_MPR_thresh_eval/", help= "the saving path for model")
parser.add_argument("--result_csv", type=str, default="./debug_MPR_thresh_eval/result_contrast.csv", help="the path for saving result")
parser.add_argument("--data_path", type=str, default="./datasets/Lastfm-360K/", help= "the data path")
parser.add_argument("--fair_reg_max", type=float, default= 10, help= "the max regulator for fairness")
parser.add_argument("--partial_ratio_s0", type=float, default= 0.5, help= "the known ratio for training sensitive attr s0 ")
parser.add_argument("--partial_ratio_s1", type=float, default= 0.1, help= "the known ratio for training sensitive attr s1 ")
parser.add_argument("--orig_unfair_model", type=str, default= "./pretrained_model/Lastfm-360K/MF_orig_model")
parser.add_argument("--gender_train_epoch", type=int, default= 1000, help="the epoch for gender classifier training")
parser.add_argument("--task_type",type=str,default="Lastfm-360K",help="Specify task type: ml-1m/tenrec/lastfm-1K/lastfm-360K")

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu_id}")
elif torch.backends.mps.is_available():
    # enable acceleration on mac chips
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Running on device: {device}")

import os
cur_path = os.getcwd()
path_to_dataset = os.path.join(cur_path,"datasets",args.task_type)
assert os.path.exists(path_to_dataset)
train_csv_path = os.path.join(path_to_dataset,"train.csv")
valid_csv_path = os.path.join(path_to_dataset,"valid.csv")
test_csv_path = os.path.join(path_to_dataset,"test.csv")
sensitive_csv_path = os.path.join(path_to_dataset,"sensitive_attribute.csv")
sensitive_csv_random_path = os.path.join(path_to_dataset,"sensitive_attribute_random.csv")


# The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = args.seed
set_random_seed(RANDOM_STATE)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Running on device: MPS")
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"Running on device: CUDA {args.gpu_id}")
else:
    device = torch.device("cpu")
    print("Running on device: CPU")

# set hyperparameters
saving_path = args.saving_path
emb_size = args.embed_size
output_size = args.output_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
evaluation_epoch = args.evaluation_epoch
weight_decay = args.weight_decay
fair_reg_max = args.fair_reg_max
beta = args.beta 


data_path = args.data_path
train_data = pd.read_csv(train_csv_path,dtype=np.int64)
valid_data = pd.read_csv(valid_csv_path,dtype=np.int64)
test_data = pd.read_csv(test_csv_path,dtype=np.int64)
orig_sensitive_attr = pd.read_csv(sensitive_csv_path,dtype=np.int64)
sensitive_attr = pd.read_csv(sensitive_csv_random_path,dtype=np.int64)
s0_known =  sensitive_attr[sensitive_attr["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_s0 * sum(sensitive_attr["gender"] == 0))]
s1_known =  sensitive_attr[sensitive_attr["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_s1 * sum(sensitive_attr["gender"] == 1))]

num_uniqueUsers = max(train_data.user_id) + 1
num_uniqueLikes = max(train_data.item_id) + 1

MF_model = matrixFactorization(np.int64(num_uniqueUsers), np.int64(num_uniqueLikes), emb_size).to(device)

if os.path.exists(args.orig_unfair_model):
    print(f"Loading pre-trained MF weights from {args.orig_unfair_model}")
    MF_model.load_state_dict(torch.load(args.orig_unfair_model, map_location=device))

print(args)

# initialized model


# # the range of priors used in Multiple Prior Guided Robust Optimization
# # here we choose 37 different priors
# resample_range = torch.tensor([
#     0.1, 0.105, 0.11, 0.12, 0.125, 0.13, 0.14, 0.15, 0.17,
#     0.18, 0.2, 0.22, 0.25, 0.29, 0.33, 0.4, 0.5, 0.67, 0.75,
#     0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 
#     0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999
# ], dtype=torch.float32).to(device)

# resample_range = torch.linspace(0.01, 0.99, 15).to(device)
resample_range = torch.tensor([
    0.1, 0.105, 0.11, 0.14, 0.15, 0.17,
    0.18, 0.2, 0.22, 0.5, 0.67, 0.75,
    0.8, 0.82, 0.84, 0.9, 0.91, 0.98, 0.985, 0.99, 0.995, 0.999
], dtype=torch.float32).to(device)

s0_ratio = args.partial_ratio_s0 
s1_ratio = args.partial_ratio_s1 

# rmse_thresh
if args.task_type == "Lastfm-360K":
    if args.seed == 1:
        rmse_thresh = 0.327087092 / 0.98
    elif args.seed == 2:
        rmse_thresh = 0.327050738 / 0.98
    elif args.seed ==3:
        rmse_thresh = 0.327054454 / 0.98
elif args.task_type == "ml-1m":
    if args.seed == 1:
        rmse_thresh = 0.412740352 / 0.98
    elif args.seed == 2:
        rmse_thresh = 0.412416265 / 0.98
    elif args.seed ==3:
        rmse_thresh = 0.412392938 / 0.98
else:
    raise ValueError("Not rmse thresh")

print("rmse thresh:" + str(rmse_thresh))

# Initialize and train amortized SST model
sst_model = AmortizedSST(emb_size).to(device)
print("[SST Classifier] Start training amortized SST classifier...")
train_amortized_sst(
    sst_model, 
    MF_model, 
    s0_known, 
    s1_known, 
    epochs=20, 
    device=device,
    alpha_max=0.6
)

print("\n[Calibration] Verifying Amortized SST Performance...")
evaluate_amortized_sst(
    sst_model, 
    MF_model, 
    s0_known, 
    s1_known, 
    device=device
)

# Pretrain MF model with Multiple Prior Robust Optimization and evaluate on validation and test set
print("[Fair MF Model] Start training fair MF model with MPR...")
val_rmse, test_rmse, best_unf, unf_test, best_epoch, best_model = \
    train_fair_mf_mpr(
        model=MF_model, 
        sst_model=sst_model,
        df_train=train_data,
        epochs=num_epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        beta=beta,
        valid_data=valid_data,
        test_data=test_data,
        resample_range=resample_range, 
        oracle_sensitive_attr=orig_sensitive_attr,
        fair_reg_max=fair_reg_max,
        s0_known=s0_known,
        s1_known=s1_known,
        device=device,
        evaluation_epoch=evaluation_epoch,
        unsqueeze=True,
        rmse_thresh=rmse_thresh
    )

os.makedirs(args.saving_path, exist_ok= True)
torch.save(MF_model.state_dict(), args.saving_path + "/MF_model")
torch.save(best_model.state_dict(), args.saving_path + "/best_model")

csv_folder = ''
for path in args.result_csv.split("/")[:-1]:
    csv_folder = os.path.join(csv_folder, path)

os.makedirs(csv_folder, exist_ok= True)

try:
    pd.read_csv(args.result_csv)
except:
    with open(args.result_csv,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["args", "val_rmse_in_that_epoch", "test_rmse_in_that_epoch", "best_unfairness_val_partial", "unfairness_test", "best_epoch"])
