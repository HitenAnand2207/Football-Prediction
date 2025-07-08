import requests
from bs4 import BeautifulSoup
from datetime import datetime

def get_today_matches():
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://www.bbc.com/sport/football/scores-fixtures/{today}"
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    matches = []
    for f in soup.find_all("article", class_="sp-c-fixture"):
        home = f.find("span", class_="sp-c-fixture__team--home").text
        away = f.find("span", class_="sp-c-fixture__team--away").text
        # maybe get link for match to fetch player stats
        matches.append((home, away))
    return matches
