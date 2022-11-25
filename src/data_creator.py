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
# %%
URL = "https://oriath.net/Audio/"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")
links = soup.find_all("a")
# %%
classes = np.array(links, dtype=object)[2:-1].tolist()
classes = [item for sublist in classes for item in sublist]
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

def match_reg(s):
    return bool(re.search(re.compile("|".join(exceptions_reg)), s))

unique_classes_idx = unique_classes.apply(match_reg)
unique_classes = list(unique_classes[~unique_classes_idx].reset_index(drop=True))

for exception in exceptions:
    unique_classes.append(exception)

unique_classes = [u_c.replace(" ", "_") for u_c in unique_classes]
unique_classes = [u_c.replace(",", "") for u_c in unique_classes]
unique_classes = [u_c.replace(":", "") for u_c in unique_classes]
unique_classes = [u_c.replace("\"", "") for u_c in unique_classes]
unique_classes = [u_c.replace("\'", "") for u_c in unique_classes]
unique_classes = [u_c.replace("-", "_") for u_c in unique_classes]
class_dict = dict(zip(unique_classes, range(len(unique_classes))))
# %%
filenames = []
for filename in tqdm(os.listdir(pathlib.Path("../Audiodata"))):
    with open(os.path.join(pathlib.Path("../Audiodata"), filename), 'r') as f:
        filenames.append(f.name.split("\\")[-1])
filenames = pd.Series(filenames)
filenames = filenames[filenames.str.contains("Conversation") == False].reset_index(drop=True).to_list()
# %%
regexes = class_dict.keys()
regex = re.compile("|".join(regexes))
classes = [re.findall(regex, filename)[0] for filename in filenames]
# %%
class_dict_pd = [class_dict[class_] for class_ in classes]
# %%
df = pd.DataFrame({
    "name":filenames,
    "class":classes,
    "class_id":class_dict_pd
})
# %%
df["name"] = df["name"].apply(lambda x:x.replace(".ogg", ".wav"))
# %%
df
# %%
def ogg2wav(ofn):
    wfn = ofn.replace('.ogg','.wav')
    x = AudioSegment.from_ogg(ofn)
    x.export(wfn, format='wav')
# %%
for filename in tqdm(os.listdir(pathlib.Path("../Audiodata"))):
    with open(os.path.join(pathlib.Path("../Audiodata"), filename), 'r') as f:
        if f.name[-4:] == ".ogg":
            ogg2wav(f.name)
            f.close()
            os.remove(f.name)
# %%
