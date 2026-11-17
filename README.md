# Satellite-Based Flood Mapping & Deforestation Monitoring (VS Code project)

Two trained U-Net (ResNet34) models:
- **Flood mapping** - trained on Sen1Floods11 (Sentinel-2 chips + flood labels)
- **Deforestation** - trained on Sentinel-2 before/after patches with Hansen forest-loss labels (collected once from Google Earth Engine), compared with an NDVI-differencing baseline

## 1. Setup (Windows / VS Code)
1. Install Python 3.10 or 3.11 and VS Code (with the Python extension).
2. Open this folder in VS Code (File > Open Folder).
3. Terminal > New Terminal, then:
```
python -m venv .venv
.venv\Scripts\activate          (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
4. Ctrl+Shift+P > "Python: Select Interpreter" > choose the `.venv` one.
5. NVIDIA GPU? Install the CUDA build of PyTorch from pytorch.org instead of the default one (otherwise training runs on CPU, which works but is slow).

## 2. Project layout
```
common.py                         shared config, datasets, model, loss, metrics, training loop
01_download_flood_data.py         downloads Sen1Floods11 into data/sen1floods11
02_train_flood.py                 trains the flood model  -> models/best_flood_unet.pt, outputs/
03_build_deforestation_dataset.py builds data/deforestation_patches.npz from Earth Engine (run once)
04_train_deforestation.py         trains the deforestation model + NDVI comparison -> models/best_change_unet.pt
05_export_samples.py              makes samples/ (used by the app and by Hugging Face)
app.py                            Gradio web app (both tabs, offline)
data/  models/  outputs/  samples/   created automatically
```

## 3. Run order
```
python 01_download_flood_data.py
python 02_train_flood.py
python 03_build_deforestation_dataset.py      # needs Earth Engine login (opens browser once)
python 04_train_deforestation.py
python 05_export_samples.py
python app.py                                 # opens http://127.0.0.1:7860
```
Earth Engine project ID: edit `GEE_PROJECT` in `common.py` (default `arj-mini`) or set the environment variable `GEE_PROJECT`.

## 4. Reusing what you already trained in Colab (skip steps)
Copy from Google Drive (`Mini_5sem`):
- `models/best_flood_unet.pt` and `models/best_change_unet.pt`  -> into `models/`
- `data/deforestation_patches.npz`                              -> into `data/`
- `data/sen1floods11.zip` -> unzip into `data/` so you get `data/sen1floods11/S2` and `data/sen1floods11/LBL`

Then you only need `05_export_samples.py` and `app.py` (and skip steps 01-04).

## 5. Demo day
Just run `python app.py` (about 10 seconds). Use a chip from `samples/flood` (dropdown) or upload any Sen1Floods11 `*_S2Hand.tif`.
