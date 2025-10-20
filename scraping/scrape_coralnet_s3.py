import getpass
from urllib.parse import urljoin, urlparse

import boto3
import numpy as np
import requests
from bs4 import BeautifulSoup
from coralnet_scraper import CoralNetDownloader


def check_s3_prefix_exists(bucket_name, s3_prefix, source_id):
    s3 = boto3.client("s3")
    prefix = f"{s3_prefix}/s{source_id}/annotations.csv"

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)

    if "Contents" in response:
        print(f"Prefix exists: {prefix}")
        return True
    else:
        print(f"Prefix does not exist: {prefix}")
        return False


# Configuration - Where to save the downloaded CoralNet images
bucket_name = "dev-datamermaid-sm-sources"
prefix = "coralnet-public-images"

# User credentials for CoralNet
username = input("CoralNet username: ")
password = getpass.getpass("CoralNet password: ")

# Initialize list of CoralNet public sources in the pyspacer-bucket
s3 = boto3.client("s3")
coralnet_bucket_name = "2310-coralnet-public-sources"

response = s3.list_objects_v2(Bucket=coralnet_bucket_name, Delimiter="/")
if "CommonPrefixes" in response:
    subdirectories = [prefix["Prefix"] for prefix in response["CommonPrefixes"]]
else:
    print("No subdirectories found.")

coralnet_sources = np.sort(
    [int(source[1:-1]) for source in subdirectories if source[0] == "s"]
).astype(str)

downloader = CoralNetDownloader(username=username, password=password)
for source_id in coralnet_sources:
    print("Source ID", source_id)
    if check_s3_prefix_exists(
        bucket_name=bucket_name, s3_prefix=prefix, source_id=source_id
    ):
        continue
    downloader.download_source(
        source_id=source_id, bucket_name=bucket_name, s3_prefix=prefix
    )

# Now check for any other public CoralNet sources not in the pyspacer-bucket - Last run: 20.10.2025
url = "https://coralnet.ucsd.edu/source/about/"

resp = requests.get(url, timeout=50)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")
anchors = soup.find_all("a", href=True)

links = sorted(
    {
        urljoin(url, a["href"])
        for a in anchors
        if urlparse(urljoin(url, a["href"])).scheme in ("http", "https")
    }
)

print("Found", len(links), "links on the page.")
source_links = [link for link in links if "/source/" in link]
print("Found", len(source_links), "links on the page.")
all_coralnet_sources = sorted({int(link.split("/")[-2]) for link in source_links})

downloader = CoralNetDownloader(username=username, password=password)
for source_id in all_coralnet_sources:
    print("Source ID", source_id)
    if check_s3_prefix_exists(
        bucket_name=bucket_name, s3_prefix=prefix, source_id=source_id
    ):
        continue
    downloader.download_source(
        source_id=source_id, bucket_name=bucket_name, s3_prefix=prefix
    )
