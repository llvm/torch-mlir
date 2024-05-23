"""Scrapes the github releases API to generate a static pip-install-able releases page.

See https://github.com/llvm/torch-mlir/issues/1374
"""

import argparse
import json

import requests

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("owner", type=str)
parser.add_argument("repo", type=str)
args = parser.parse_args()

# Get releases
response = requests.get(
    f"https://api.github.com/repos/{args.owner}/{args.repo}/releases"
)
body = json.loads(response.content)

# Parse releases
releases = []
for row in body:
    for asset in row["assets"]:
        releases.append((asset["name"], asset["browser_download_url"]))

# Output HTML
html = """<!DOCTYPE html>
<html>
  <body>
"""
for name, url in releases:
    html += f"    <a href='{url}'>{name}</a><br />\n"
html += """  </body>
</html>"""
print(html)
