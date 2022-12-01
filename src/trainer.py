# %%
import torch
from torch import nn
from model.AudioClassifier import AudioClassifier
import pandas as pd
from sound_dataset import SoundDataset
from torch import utils
from tqdm import tqdm


# %%
def training(model, train_dl, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_pred = 0
        for i, data in enumerate(tqdm(train_dl)):
            inputs, lables = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)

            correct_prediction += (prediction == lables).sum().item()
            total_pred += prediction.shape[0]
        
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_pred
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    print("Finished Training")
# %%
if __name__ == "__main__":
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    train_ds = SoundDataset(df_train, "../Audiodata/")
    test_ds = SoundDataset(df_test, "../Audiodata/")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    myModel = AudioClassifier()
    device = torch.device("cuda:0" if torch.cuda.is_available()else "cpu")
    myModel = myModel.to(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    #%%
    num_epochs = 20
    training(myModel, train_dl, num_epochs, device)
    # %%
