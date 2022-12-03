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
from sklearn.model_selection import train_test_split


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


def train_test_val_split(train_size, test_size, val_size):
    df_tmp, df_test = train_test_split(df, test_size=test_size)
    df_train, df_val = train_test_split(df_tmp, test_size=val_size/(1-test_size))
    return df_train, df_test, df_val


if __name__ == "__main__":
    url = "https://oriath.net/Audio/"
    links = get_links(url)
    unique_classes, class_dict = get_classes(links)
    df = build_dataframe(unique_classes, class_dict)
    #transform_audio_data()

    train_size, test_size, val_size = 0.7, 0.15, 0.15
    df_train, df_test, df_val = train_test_val_split(train_size, test_size, val_size)
    df_train.to_csv("train.csv")
    df_test.to_csv("test.csv")
    df_val.to_csv("val.csv")
    print(df_train.shape, df_test.shape, df_val.shape)

# %%
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_val = pd.read_csv("val.csv")

all_data = pd.concat([df_train, df_test, df_val]).reset_index(drop = True)
#
split_data = all_data.groupby("class_id").class_id.value_counts().nlargest(100)
df_count = pd.DataFrame({
    "class_id":np.array(split_data.index.to_list())[:,1],
    "count":split_data.to_list()
})
df_count["name"] = df_count.class_id.map({v: k for k, v in class_dict.items()})
print(df_count)

