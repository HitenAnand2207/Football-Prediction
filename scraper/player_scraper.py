import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_player_stats_and_scorers(match_url):
    resp = requests.get(match_url, headers={"User-Agent":"Mozilla/5.0"})
    soup = BeautifulSoup(resp.content, "html.parser")
    # Box stats table
    tables = pd.read_html(resp.text)
    stats = tables[0]  # adjust idx based on page
    # Find scorers in the HTML
    scorers = []
    for tag in soup.find_all("a", href=True):
        if 'goal-scorer' in tag.get('href'):
            scorers.append(tag.text)
    return stats, scorers
