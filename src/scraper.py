# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pydub import AudioSegment
# %%
URL = "https://oriath.net/Audio/"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
# %%
links = soup.find_all("a")
print(links)
# %%
pathlib.Path('../Audiodata').mkdir(parents=True, exist_ok=True) 
# %%
def scrape(link):
    child_page = requests.get(URL+link["href"])
    child_soup = BeautifulSoup(child_page.content, "html.parser")
    child_links = child_soup.find_all("a", class_="play")
    for i, child_link in enumerate(child_links):
        response = requests.get(URL[:-7]+child_link["href"])
        l = link["href"].replace("/", "_")
        open(f"../Audiodata/{l}{i}.ogg" , "wb" ).write(response.content)
# %%
with ThreadPoolExecutor(max_workers=20) as executor:
    for j, link in enumerate(links[2:]):
        future = executor.submit(scrape, link)
# %%
for filename in tqdm(os.listdir(pathlib.Path("../Audiodata"))):
    with open(os.path.join(pathlib.Path("../Audiodata"), filename), 'r') as f:
        path = pathlib.Path(f.name).resolve()
        audio = AudioSegment.from_wav(path)
        
        six_sec = 6 * 1000
    
        divider = (int) (audio.duration_seconds * 1000 / six_sec)
        if audio.duration_seconds > 6:
            for i in range(divider):
                audio[i * six_sec: (i + 1) * six_sec].export(f"../Audiodata/{f.name[:-4]}_{i}.wav", format = "wav")
                f.close()
                if (i == divider-1) and ((int)(divider * (six_sec / 1000) != audio.duration_seconds)):
                    audio[(i+1) * six_sec:].export(f"../Audiodata/{f.name[:-4]}_{i+1}.wav", format = "wav")
                    f.close()
                   
            os.remove(path)
# %%
