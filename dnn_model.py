from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame

from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm
import argparse

from feature_engineering import load_data, full_feature_cols, load_data_predict
from pytorch_tools import set_seed, save_model, clean_models, find_best_model_file, EarlyStopping, dump_data_info, load_data_info
from stock_utils import stockids_training


_device = "cuda:0"
_lr = 0.001
_min_lr = 0.0000001
_max_epoch = 80000
_batch_size = 8192
_model_name = 'dnn_model'
_models_folder = 'models'

_input_feature_cols = ['p_open_s', 'p_close_s', 'p_high_s', 'p_low_s', 'p_volume_s']
_output_feature_cols = ['n_high_s', 'n_low_s']
_feature_info_cols = ['stockid']
_output_size = len(_output_feature_cols)
_input_size = len(_input_feature_cols)
_encoder_hidden_size = 64
_decoder_hidden_size = 32
_encoder_layers = 3
_decoder_layers = 3
_p_steps = 44
_n_steps = 2
_lr_patience = 50
_stop_patience = 800

# Dataset
class StockDataset(Dataset):
    def __init__(self, data: type(DataFrame), feautre_cols: List[str], label_cols: List[str], p_steps: int, n_steps: int, train: bool = True):
        feautre_cols = full_feature_cols(feautre_cols, p_steps)
        if train:
            label_cols = full_feature_cols(label_cols, n_steps)

        self.x = data[feautre_cols].values
        self.y = data[label_cols].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def data_to_tensor(inputs, device):
    return torch.tensor(inputs).float().to(device=device)


# Model
class TimeSeriesModel_NStep(nn.Module):
    def __init__(self, output_size, input_size, encoder_hidden_size, decoder_hidden_size, n_steps, encoder_layers, decoder_layers):
        super(TimeSeriesModel_NStep, self).__init__()
        
        self.dropout = nn.Dropout(p=0.2)
        self.output_size = output_size
        self.input_size = input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.n_steps = n_steps
        
        self.fc_base_status = nn.Linear(input_size, encoder_hidden_size)
        self.encoder = nn.GRU(input_size=input_size, hidden_size=encoder_hidden_size,
                            num_layers=encoder_layers, batch_first=True, dropout=0.2)
        self.fc_encoder = nn.Linear(encoder_hidden_size * encoder_layers, decoder_hidden_size)         
        
        self.decoder = nn.GRU(input_size=decoder_hidden_size, hidden_size=decoder_hidden_size,
                            num_layers=decoder_layers, batch_first=True, dropout=0.2)
        
        self.fc = nn.Linear(decoder_hidden_size, output_size)

    def forward(self, x, seq_length):
        # Propagate input through GRU
        x = x.view((-1, seq_length, self.input_size))
        base_status = self.fc_base_status(x.clone().detach()[:, 0, :])
        base_status = base_status.repeat((self.encoder_layers, 1)).reshape((self.encoder_layers, -1, self.encoder_hidden_size))

        _, encoder_hidden = self.encoder(x, base_status) 
        encoder_hidder = encoder_hidden.transpose(0, 1).reshape((-1, self.encoder_hidden_size * self.encoder_layers))
        encode_status = self.fc_encoder(encoder_hidder)
        #encode_status = F.relu(encode_status)

        decoder_input = encode_status.repeat((1, 1, self.n_steps))
        decoder_input = decoder_input.view(-1, self.n_steps, self.decoder_hidden_size)
        decoder_status = encode_status.repeat((self.decoder_layers, 1)).reshape((self.decoder_layers, -1, self.decoder_hidden_size))

        decoder_out, _ = self.decoder(decoder_input, decoder_status) 
        decoder_out = decoder_out.reshape(-1, self.decoder_hidden_size)
        #decoder_out = F.relu(decoder_out)

        out = self.fc(self.dropout(decoder_out))
        out = out.reshape(-1, self.output_size * self.n_steps)
        
        return out

def init_model():
    return TimeSeriesModel_NStep(_output_size, _input_size, _encoder_hidden_size, _decoder_hidden_size, _n_steps, _encoder_layers, _decoder_layers).to(device=_device)

# Train
def train(model, stock_ids, model_name = _model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
    criterion = torch.nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=_lr_patience, verbose=True, cooldown=1, min_lr=_min_lr, eps=_min_lr)
    earlyStop = EarlyStopping(model_name, _models_folder, patience=_stop_patience)
    

    data, features_info = load_data(stock_ids, _p_steps)
    dump_data_info(model_name, _models_folder, features_info)

    train_dataset = StockDataset(data[data['day'] < '2018-01-01'], _input_feature_cols, _output_feature_cols, _p_steps, _n_steps)
    valid_dataset = StockDataset(data[data['day'] >= '2018-01-01'], _input_feature_cols, _output_feature_cols, _p_steps, _n_steps)

    clean_models(model_name, _models_folder)
    pbar = tqdm(range(0, _max_epoch))
    
    for _ in pbar:
        train_dataloader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True, num_workers=0)
        valid_dataloader = DataLoader(valid_dataset, batch_size=_batch_size, shuffle=True, num_workers=0)
        optimizer.zero_grad()
        model.train()
        
        total_train_loss = []
        for _, items in enumerate(train_dataloader):
            x = data_to_tensor(items[0], _device)
            y = data_to_tensor(items[1], _device)
            train_outputs = model(x, _p_steps)

            train_loss = criterion(train_outputs, y)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
            optimizer.step()
            total_train_loss.append(train_loss)
           
        train_loss = torch.mean(torch.stack(total_train_loss))

        total_valid_loss = []
        with torch.no_grad():
            model.eval()
            for _, items in enumerate(valid_dataloader):
                x = data_to_tensor(items[0], _device)
                y = data_to_tensor(items[1], _device)
                valid_outputs = model(x, _p_steps)
                validate_loss = criterion(valid_outputs, y)
                total_valid_loss.append(validate_loss)
        validate_loss = torch.mean(torch.stack(total_valid_loss))

        earlyStop(validate_loss, model)
        if earlyStop.early_stop:
            break

        scheduler.step(train_loss)
        pbar.set_description("{0:.6f}, {1:.6f}".format(train_loss, validate_loss))
    
    return model

def load_dnn_model(model_name = _model_name):
    model = init_model()
    best_model_parameters = find_best_model_file(model_name, _models_folder)
    model.load_state_dict(torch.load(best_model_parameters))

    return model

def predict(stock_ids, model_name = _model_name):
    data_info = load_data_info(_model_name, _models_folder)
    data = load_data_predict(stock_ids, data_info, _p_steps)
    feautre_cols = full_feature_cols(_input_feature_cols, _p_steps)

    model = load_dnn_model()
    with torch.no_grad():
        model.eval()
        x = data_to_tensor(data[feautre_cols].values, _device)
        outputs = model(x, _p_steps).cpu()

        for row in range(0, len(stock_ids)):
            stats = data_info[stock_ids[row]]
            outputs[row][0] = outputs[row][0] * stats['high_std'] + stats['high_mean']
            outputs[row][2] = outputs[row][2] * stats['high_std'] + stats['high_mean']
            outputs[row][1] = outputs[row][1] * stats['low_std'] + stats['low_mean']
            outputs[row][3] = outputs[row][3] * stats['low_std'] + stats['low_mean']

            print("{0}: {1}".format(stock_ids[row], outputs[row]))

        return outputs
        

print(predict(stockids_training))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-stockids', nargs='+')
    parser.add_argument('-train', dest='train_flag', action='store_true')
    parser.set_defaults(train_flag=False)
    
    args = parser.parse_args()
    stockids = stockids_training if not args.stockids else args.stockids

    result = {}
    model = init_model()
    if args.train_flag:
        train(model, stockids)

    predict(stockids)