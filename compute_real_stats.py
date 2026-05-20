import os
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

BASE_DIR = r"c:\Users\asmi\Desktop\Projects\URL Phishing\SafeWeb-AI-Fake-URL-Detection-Using-Machine-Learning"

df = pd.read_csv(os.path.join(BASE_DIR, 'phishing_site_urls.csv'))

df['Label'] = df['Label'].map({'good': 0, 'bad': 1})
df_sample = df.sample(n=10000, random_state=42).copy().reset_index(drop=True)

_IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

def extract_features(url: str) -> list:
    url_lower = url.lower()
    if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
        url = "http://" + url
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    domain = parsed.netloc
    return [
        len(url),
        len(domain),
        url.count("."),
        url.count("-"),
        url.count("@"),
        url.count("/"),
        sum(c.isdigit() for c in url),
        1 if parsed.scheme.lower() == "https" else 0,
        1 if _IP_PATTERN.match(hostname or domain) else 0,
    ]

features_list = []
for url in df_sample['URL']:
    try:
        features_list.append(extract_features(url))
    except Exception:
        features_list.append([0]*9)

features_df = pd.DataFrame(features_list, columns=[
    "url_len", "dom_len", "dots", "hyphens", "at_sym", "slashes", "digits", "https", "ip"
])
features_df['Label'] = df_sample['Label']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n--- Statistics for Legitimate (Label = 0) ---")
print(features_df[features_df['Label'] == 0].describe().loc[['mean', 'std']])

print("\n--- Statistics for Phishing (Label = 1) ---")
print(features_df[features_df['Label'] == 1].describe().loc[['mean', 'std']])
