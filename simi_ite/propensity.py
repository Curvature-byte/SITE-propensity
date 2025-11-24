import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train_torch_model(model, x, t, epochs=100, lr=0.01):
    n_treat = np.sum(t == 1)
    n_control = len(t) - n_treat
    pos_weight = torch.FloatTensor([float(n_control) / n_treat if n_treat > 0 else 1.0])
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_tensor = torch.FloatTensor(x)
    t_tensor = torch.FloatTensor(t).reshape(-1, 1)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = criterion(output, t_tensor)
        loss.backward()
        optimizer.step()
    return model

def get_predictions(model, x):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x)
        logits = model(x_tensor)
        prob_1 = torch.sigmoid(logits).numpy()
        prob_0 = 1.0 - prob_1
        prob_all = np.hstack([prob_0, prob_1])
        pred = (prob_1 > 0.5).astype(int).flatten()
    return pred, prob_all

def propensity_score_training(data, label, mode):

    '''
    :param data: pre-treatment covariates
    :param label: treatment that the units accually took
    :param mode: the method to to get the propsensity score
    :return: the propensity socre (the probability that a unit is in the treated group); the trainied propensity calcualtion model
    '''

    train_x, eva_x, train_t, eva_t = train_test_split(data, label, test_size=0.3, random_state=42)
    
    input_dim = data.shape[1]
    
    if mode == 'Logistic-regression':
        model = nn.Sequential(nn.Linear(input_dim, 1))
        model = train_torch_model(model, train_x, train_t)

        pred_eva, prob_eva = get_predictions(model, eva_x)
        pred_train, prob_train = get_predictions(model, train_x)
        
        acc_train = accuracy_score(train_t, pred_train)
        f1_train = f1_score(train_t, pred_train)
        f1_eva = f1_score(eva_t, pred_eva)

        acc_eva = accuracy_score(eva_t, pred_eva)

        result_all, prob_all = get_predictions(model, data)

        return prob_all, model
        
    if mode == 'SVM':
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        model = train_torch_model(model, train_x, train_t)
        
        pred_eva, prob_eva = get_predictions(model, eva_x)
        pred_train, prob_train = get_predictions(model, train_x)
        
        acc_train = accuracy_score(train_t, pred_train)
        f1_train = f1_score(train_t, pred_train)
        f1_eva = f1_score(eva_t, pred_eva)

        acc_eva = accuracy_score(eva_t, pred_eva)
        print(acc_train)
        print(acc_eva)
        print(f1_train)
        print(f1_eva)
        
        result_all, prob_all = get_predictions(model, data)
        print(result_all[1:10])
        print(prob_all[1:10,:])
        return prob_all, model

    if mode == 'CART':
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        model = train_torch_model(model, train_x, train_t)
        
        pred_eva, pred_eva_prob = get_predictions(model, eva_x)

        f1_eva = f1_score(eva_t, pred_eva)
        acc_eva = accuracy_score(eva_t, pred_eva)
        print(pred_eva_prob)
        print(acc_eva)
        print(f1_eva)

def onehot_trans(t, catog):
    trans = np.zeros([t.shape[0], catog.size])
    for i in range(t.shape[0]):
        if t[i,0] == 0:
            trans[i,0] = 1
        else:
            trans[i,1] = 1
    return trans

def load_propensity_score(model_file_name, x):
    loaded_model = torch.load(model_file_name)
    loaded_model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x)
        result = torch.sigmoid(loaded_model(x_tensor)).numpy()
    propensity_score = result.flatten()
    return propensity_score