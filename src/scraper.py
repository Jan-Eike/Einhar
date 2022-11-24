# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://oriath.net/Audio/"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
# %%
links = soup.find_all("a")
print(links)
# %%
for link in links[2:]:
    link_url = link["href"]
    print(link_url)

# %%
links[2]["href"]
# %%
page2 = requests.get(URL+links[2]["href"])
# %%
soup2 = BeautifulSoup(page2.content, "html.parser")
# %%
links = soup2.find_all("a", class_="play")
print(links)
# %%
print(URL[:-7]+links[2]["href"])
# %%
response = requests.get(URL[:-7]+links[2]["href"])
open("marauder.ogg", "wb").write(response.content)
# %%
pd.DataFrame({
    "file_name":"",
    "class":"marauder",
    "class_id":0
})
# %%