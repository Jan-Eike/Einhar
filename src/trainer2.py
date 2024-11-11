# %%
import torch
from torch import nn
from model.AudioClassifier import AudioClassifier
import pandas as pd
from sound_dataset import SoundDataset
from tqdm import tqdm
import pathlib
import numpy as np
from audio_preprocessing import AudioPreprocessing


# %%
def training(model, train_dl, val_dl, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1)
    min_valid_loss = np.inf
    for epoch in range(num_epochs):
        print(optimizer)
        train_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        model.train()
        for i, data in enumerate(tqdm(train_dl)):
            inputs, lables = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = nn.functional.nll_loss(outputs.squeeze(), lables)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
        
        valid_loss = 0.0
        model.eval()
        for i, data in enumerate(tqdm(val_dl)):
            inputs, lables = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = nn.functional.nll_loss(outputs.squeeze(), lables)
            valid_loss += loss.item()

            # val accuracy
            _, prediction = torch.max(outputs.squeeze(), 1)
            correct_prediction += (prediction == lables).sum().item()
            total_prediction += prediction.shape[0]

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dl)} \t\t Validation Loss: {valid_loss / len(val_dl)} \t\t Validation Accuracy: {correct_prediction / total_prediction}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), './model/saved_models/model_1.pth')


    print("Finished Training")
# %%
if __name__ == "__main__":
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    df_val = pd.read_csv("val.csv")

    train_ds = SoundDataset(df_train, "../Audiodata/")
    test_ds = SoundDataset(df_test, "../Audiodata/")
    val_ds = SoundDataset(df_val, "../Audiodata/")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)

    model = AudioClassifier()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    #%%
    pathlib.Path('./model/saved_models').mkdir(parents=True, exist_ok=True) 
    num_epochs = 100
    training(model, train_dl, val_dl, num_epochs, device)
    # %%
