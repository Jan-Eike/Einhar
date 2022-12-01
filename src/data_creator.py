# %%
import pandas as pd
import re
import os
import pathlib
import requests
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import torchaudio
from audio_utils import AudioUtils
from torch.utils.data import random_split
from sound_dataset import SoundDataset
import torch


def get_links(url):
    """get all "a" links from the given url"""
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    links = soup.find_all("a")
    return links


def flatten(l):
    """flatten list"""
    return [item for sublist in l for item in sublist]


def remove_special_chars(unique_classes):
    # replace special characters the same way they are replaced on the website
    unique_classes = [u_c.replace(" ", "_") for u_c in unique_classes]
    unique_classes = [u_c.replace(",", "") for u_c in unique_classes]
    unique_classes = [u_c.replace(":", "") for u_c in unique_classes]
    unique_classes = [u_c.replace("\"", "") for u_c in unique_classes]
    unique_classes = [u_c.replace("\'", "") for u_c in unique_classes]
    unique_classes = [u_c.replace("-", "_") for u_c in unique_classes]
    return unique_classes


def match_reg(s, exceptions_reg):
        return bool(re.search(re.compile("|".join(exceptions_reg)), s))


def get_classes(links):
    """get class names and assign id"""
    classes = np.array(links, dtype=object)[2:-1].tolist()
    classes = flatten(classes)
    unique_classes = list(set(classes))
    unique_classes = pd.Series(unique_classes)
    unique_classes = unique_classes[unique_classes.str.contains("Conversation") == False].reset_index(drop=True)
    exceptions = [
        "Malachai",
        "Maligaro",
        "Shavronne",
        "Sirus",
        "Veritania",
        "Drox",
        "Al-Hezmin",
        "Baran",
        "Kirac"
    ]
    exceptions_reg = ["[\w]*"+item+"[\w]*" for item in exceptions]

    unique_classes_idx = unique_classes.apply(match_reg, exceptions_reg=exceptions_reg)
    unique_classes = list(unique_classes[~unique_classes_idx].reset_index(drop=True))
    unique_classes = unique_classes + [exception for exception in exceptions]
    unique_classes = remove_special_chars(unique_classes)
    # assign a number to each class name
    class_dict = dict(zip(unique_classes, range(len(unique_classes))))
    print(len(unique_classes))
    return unique_classes, class_dict


def get_filenames():
    filenames = []
    for filename in tqdm(os.listdir(pathlib.Path("../Audiodata"))):
        with open(os.path.join(pathlib.Path("../Audiodata"), filename), 'r') as f:
            filenames.append(f.name.split("\\")[-1])
    filenames = pd.Series(filenames)
    filenames = filenames[filenames.str.contains("Conversation") == False].reset_index(drop=True).to_list()
    return filenames


def build_dataframe(unique_classes, class_dict):
    filenames = get_filenames()
    regexes = class_dict.keys()
    regex = re.compile("|".join(regexes))
    classes = [re.findall(regex, filename)[0] for filename in filenames]
    class_dict_pd = [class_dict[class_] for class_ in classes]
    df = pd.DataFrame({
        "name":filenames,
        "class":classes,
        "class_id":class_dict_pd
    })
    df["name"] = df["name"].apply(lambda x:x.replace(".ogg", ".wav"))
    return df


def ogg2wav(ofn):
    wfn = ofn.replace('.ogg','.wav')
    x = AudioSegment.from_ogg(ofn)
    x.export(wfn, format='wav')


def transform_audio_data():
    for filename in tqdm(os.listdir(pathlib.Path("../Audiodata"))):
        with open(os.path.join(pathlib.Path("../Audiodata"), filename), 'r') as f:
            if f.name[-4:] == ".ogg":
                ogg2wav(f.name)
                f.close()
                os.remove(f.name)


if __name__ == "__main__":
    url = "https://oriath.net/Audio/"
    links = get_links(url)
    unique_classes, class_dict = get_classes(links)
    df = build_dataframe(unique_classes, class_dict)
    transform_audio_data()

    train_temp = df.sample(frac= 0.8)
    train_csv = df.sample(frac= 0.8).to_csv("train.csv")
    test_csv = df.drop(train_temp.index).to_csv("test.csv")

    # print one example
    with open(os.path.join(pathlib.Path("../Audiodata"), "Act_1_Bestel_0.wav"), 'r') as f:
        path = pathlib.Path(f.name).resolve()
        audio_file = torchaudio.load(path)

    AudioUtils.print_stats(*audio_file)
    AudioUtils.plot_waveform(*audio_file)
    AudioUtils.plot_specgram(*audio_file)
