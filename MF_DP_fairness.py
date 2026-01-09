import argparse
import numpy as np
import pandas as pd
from evaluation import validate_fairness, test_fairness
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from collaborative_models import matrixFactorization
import copy

parser = argparse.ArgumentParser(description='fairRec')
parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
parser.add_argument("--embed_size", type=int, default= 64, help= "the embedding size of MF")
parser.add_argument("--output_size", type=int, default= 1, help="the output size of MF")
parser.add_argument("--num_epochs", type=int, default= 200, help= "the max epoch of training")
parser.add_argument("--learning_rate", type= float, default= 1e-3, help="the learning rate for MF model")
parser.add_argument("--batch_size", type= int, default= 32768, help= "the batchsize for training")
parser.add_argument("--evaluation_epoch", type= int, default= 3, help= "the evaluation epoch")
parser.add_argument("--weight_decay", type= float, default= 1e-7, help= "the weight_decay for training")
parser.add_argument('--seed', type=int, default=1, help="the random seed")
parser.add_argument("--saving_path", type=str, default= "./orig_MF_temp", help= "the saving path for model")
parser.add_argument("--result_csv", type=str, default="./orig_MF_temp/result.csv", help="the path for saving result")
parser.add_argument("--data_path", type=str, default="./datasets/ml-1m/", help= "the data path")
parser.add_argument("--fair_reg", type=float, default= 0, help= "the regulator for fairness, when fair_reg equals to 0, means MF without fairness regulation")
parser.add_argument("--partial_ratio_s0", type=float, default= 1, help= "the known ratio for training sensitive attr s0 ")
parser.add_argument("--partial_ratio_s1", type=float, default= 1, help= "the known ratio for training sensitive attr s1 ")
parser.add_argument("--task_type",type=str,default="ml-1m",help="Specify task type: ml-1m/tenrec/Lastfm(Lastfm-1K)/Lastfm-360K")


args = parser.parse_args()

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = args.seed
set_random_seed(RANDOM_STATE)
device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
# set hyperparameters
saving_path = args.saving_path
emb_size = args.embed_size
output_size = args.output_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
evaluation_epoch = args.evaluation_epoch
weight_decay = args.weight_decay
fair_reg = args.fair_reg
task_type = args.task_type 

def validate_fairness_rmse(model, df_train, epochs, lr, weight_decay, batch_size, valid_data, test_data, sensitive_attr, fair_reg, s0_known, s1_known, evaluation_epoch=10, unsqueeze=False, shuffle=True):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    best_val_rmse = 100
    test_rmse_in_that_epoch = 0
    best_epoch = 0
    naive_unfairness_val_in_that_epoch = 0
    naive_unfairness_test_in_that_epoch = 0
    for idx in range(epochs):
        j = 0
        loss_total = 0
        fair_reg_total = 0
        random_id = np.array([i for i in range(len(df_train))])
        if shuffle:
            np.random.shuffle(random_id)
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size):
            # data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            data_batch = df_train.loc[random_id[batch_i:(batch_i+batch_size)]].reset_index(drop=True)
            #train_user_input, train_item_input, train_ratings = get_instances_with_neg_samples(data_batch, probabilities, num_negatives,device)
            # train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_uniqueLikes, num_negatives,device)
            train_ratings = torch.FloatTensor(np.array(data_batch["label"])).to(device)
            train_user_input = torch.LongTensor(np.array(data_batch["user_id"])).to(device)
            train_item_input = torch.LongTensor(np.array(data_batch["item_id"])).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat, train_ratings.view(-1))
            loss_total += loss.item()
                 
            # fairness regulation
            # partial s1 average pred:
            s1_known = np.isin(data_batch["user_id"], s1_known)
            s1_known_pred_mean = y_hat[s1_known].mean()

            # partial s0 average pred:
            s0_known = np.isin(data_batch["user_id"], s0_known)
            s0_known_pred_mean = y_hat[s0_known].mean()

            # if no s0 or s1, then the regulation is set to 0
            if sum(s1_known) * sum(s0_known) != 0:
                fair_regulation = torch.abs(s1_known_pred_mean - s0_known_pred_mean) * fair_reg
            else:
                fair_regulation = torch.tensor(0)

            fair_reg_total += fair_regulation.item()
            loss = loss + fair_regulation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            j = j+1
        print('epoch: ', idx, 'average loss: ',loss_total/ j, "fair reg:", fair_reg_total/j)
        if idx % evaluation_epoch == 0 :
            rmse_val, naive_unfairness_val = validate_fairness(model, valid_data, sensitive_attr, s0_known, s1_known, device)
            rmse_test, naive_unfairness_test = test_fairness(model, test_data, sensitive_attr, device)
            print('epoch: ', idx, 'validation rmse:', rmse_val, 'Unfairness:', naive_unfairness_val)
            print('epoch: ', idx, 'test rmse:', rmse_test, "Unfairness:", naive_unfairness_test)

            if rmse_val < best_val_rmse:
                best_val_rmse = rmse_val
                test_rmse_in_that_epoch = rmse_test
                best_epoch = idx
                naive_unfairness_val_in_that_epoch = naive_unfairness_val
                naive_unfairness_test_in_that_epoch = naive_unfairness_test
                best_model = copy.deepcopy(model)

    return best_val_rmse, test_rmse_in_that_epoch, naive_unfairness_val_in_that_epoch, naive_unfairness_test_in_that_epoch, best_epoch, best_model


# load data
# model = MF_model
# df_val = valid_data
# df_sensitive_attr = sensitive_attr

data_path = args.data_path
train_data = pd.read_csv(data_path + "train.csv",dtype=np.int64)
valid_data = pd.read_csv(data_path + "valid.csv",dtype=np.int64)
test_data = pd.read_csv(data_path + "test.csv",dtype=np.int64)
sensitive_attr = pd.read_csv(data_path + "sensitive_attribute.csv",dtype=np.int64)
random_sens_attr = pd.read_csv(data_path + "sensitive_attribute_random.csv",dtype=np.int64)

# generating sensitive attr mask from shuffled gender list
s0_known =  random_sens_attr[random_sens_attr["gender"] == 0]["user_id"].to_numpy()[: int(args.partial_ratio_s0 * sum(random_sens_attr["gender"] == 0))]
s1_known =  random_sens_attr[random_sens_attr["gender"] == 1]["user_id"].to_numpy()[: int(args.partial_ratio_s1 * sum(random_sens_attr["gender"] == 1))]

num_uniqueUsers = max(train_data.user_id) + 1
# num_uniqueLikes = len(train_data.like_id.unique())
num_uniqueLikes = max(train_data.item_id) + 1
# start training the NCF model
print(int(num_uniqueLikes))
print(int(num_uniqueUsers))
MF_model = matrixFactorization(np.int64(num_uniqueUsers), np.int64(num_uniqueLikes), emb_size).to(device)


# model = MF_model
# df_train = train_data
# epochs = num_epochs
# lr = learning_rate
# batch_size = batch_size
# # num_negatives = num_negatives
# unsqueeze=True

best_val_rmse, test_rmse_in_that_epoch, unfairness_val, unfairness_test, best_epoch, best_model = \
        validate_fairness_rmse(MF_model,train_data,num_epochs,learning_rate, weight_decay, batch_size, valid_data, \
            test_data, sensitive_attr, fair_reg, s0_known, s1_known, evaluation_epoch= evaluation_epoch, unsqueeze=True)

os.makedirs(args.saving_path, exist_ok= True)
torch.save(best_model.state_dict(), args.saving_path + "/MF_orig_model")

csv_folder = ''
for path in args.result_csv.split("/")[:-1]:
    csv_folder = os.path.join(csv_folder, path)

os.makedirs(csv_folder, exist_ok= True)

try:
    pd.read_csv(args.result_csv)
except:
    with open(args.result_csv,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["args", "best_val_rmse", "test_rmse_in_that_epoch", "unfairness_val_partial", "unfairness_test", "best_epoch"])

with open(args.result_csv,"a") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow([args, best_val_rmse, test_rmse_in_that_epoch, unfairness_val, unfairness_test, best_epoch])
