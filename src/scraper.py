# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

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
number_links = len(links)
data_dir = 'Audiodata'
print(number_links)
print(os.listdir(data_dir))
# %%
print(URL[:-7]+links[2]["href"])
filenames = []
for i in range(0,number_links):
    filenames.append(os.listdir(data_dir)[i])
print(filenames)
# %%

for i in range(0,number_links):
    response = requests.get(URL[:-7]+links[0]["href"])
    open("./Audiodata/maurauder"+ str(i)+ ".ogg" , "wb" ).write(response.content)
    

#response = requests.get(URL[:-7]+links[2]["href"])
#open("marauder.ogg", "wb").write(response.content)

# %%
data = pd.DataFrame({
    
    "file_name":filenames,
    "class":"marauder",
    "class_id":0
})
print(data)

# %%