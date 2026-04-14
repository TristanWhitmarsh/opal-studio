import json
import os
import urllib.request
import zipfile
import io

bioimageio_path = r"C:\Users\Tristan\miniconda3\envs\opal-env\Lib\site-packages\instanseg\bioimageio_models"
os.makedirs(bioimageio_path, exist_ok=True)

url = 'https://api.github.com/repos/instanseg/instanseg/releases'
data = json.loads(urllib.request.urlopen(url).read())

for rel in data:
    for m in rel.get('assets', []):
        name = m.get('name')
        if not name.endswith('.zip'): continue
        
        folder_name = name.replace('.zip', '')
        # Only download if missing
        if os.path.exists(os.path.join(bioimageio_path, folder_name, "instanseg.pt")):
            continue
            
        print(f"Downloading {name}...")
        download_url = m.get('browser_download_url')
        response = urllib.request.urlopen(download_url)
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            out_path = os.path.join(bioimageio_path, folder_name)
            os.makedirs(out_path, exist_ok=True)
            z.extractall(out_path)
            # sometimes the zip contains the inner folder
            # let's check
            extracted = os.listdir(out_path)
            if "instanseg.pt" not in extracted:
               for item in extracted:
                   if os.path.isdir(os.path.join(out_path, item)) and "instanseg.pt" in os.listdir(os.path.join(out_path, item)):
                       # move up
                       import shutil
                       for f in os.listdir(os.path.join(out_path, item)):
                           shutil.move(os.path.join(out_path, item, f), out_path)
                       break
        print(f"Extracted {folder_name} to {out_path}")
