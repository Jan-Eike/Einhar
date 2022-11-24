# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

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
data = pd.DataFrame({
    
    "file_name":filenames,
    "class":"marauder",
    "class_id":0
})
print(data)
# %%