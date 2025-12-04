import os
import gdown

models = {
    "rf_model.joblib": "https://drive.google.com/uc?id=1Va6lxlHMO-coFtIjdmaSM9zHFdMQ1NWZ",
    "scaler.joblib": "https://drive.google.com/uc?id=1p9ASCYCzka2SJO1ET1214Wl9u5QClA48"
}

os.makedirs("models", exist_ok=True)

for filename, url in models.items():
    path = os.path.join("models", filename)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        gdown.download(url, path, quiet=False)
    else:
        print(f"{filename} already exists.")
