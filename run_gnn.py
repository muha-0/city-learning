"""
run_gnn.py — replicates the gnn.ipynb notebook flow using the modularised src/.

Usage (from the city_learning directory):
    python run_gnn.py

Or from the repo root:
    python -m modules.city_learning.run_gnn
"""
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]   # mapedia-master/
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Config  (mirrors the notebook constants)
# ---------------------------------------------------------------------------
SEED      = 42
EPOCHS    = 10_000
P_MASK    = 0.30
EVAL_EVERY = 50

# Jakarta bounding box
MIN_LAT, MAX_LAT = -6.3725962, -6.0785515
MIN_LON, MAX_LON = 106.686105, 106.9737509

# PostgreSQL connection
DB_HOST = "cs-u-spatial-406.cs.umn.edu"
DB_NAME = "gis"
DB_USER = "gis"
DB_PASS = "gis"
DB_PORT = 5432

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "jakarta_gat_multitask.pt"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------------------------------------------------------
# Cell 0 — Load graph from DB
# ---------------------------------------------------------------------------
from src.data.graph_loader import load_graph_from_db

G = load_graph_from_db(
    min_lat=MIN_LAT, max_lat=MAX_LAT,
    min_lon=MIN_LON, max_lon=MAX_LON,
    repo_root=REPO_ROOT,
)

# ---------------------------------------------------------------------------
# Cell 1 — Build edge GeoDataFrame + merge road attributes
# ---------------------------------------------------------------------------
from src.data.road_attributes import build_final_gdf

final_gdf = build_final_gdf(
    G,
    db_host=DB_HOST, db_name=DB_NAME,
    db_user=DB_USER, db_pass=DB_PASS,
    db_port=DB_PORT,
)

# ---------------------------------------------------------------------------
# Cell 2 — Prepare dataset (split, normalise, build Data objects)
# ---------------------------------------------------------------------------
from src.training.dataset import prepare_dataset

ds = prepare_dataset(
    final_gdf,
    seed=SEED,
    train_frac=0.80,
    val_frac=0.10,
    test_frac=0.10,
    grid_size_m=1000.0,
    utm_epsg=32748,     # Jakarta: UTM zone 48S
    device=device,
)

data_train      = ds["data_train"]
data_val        = ds["data_val"]
data_test       = ds["data_test"]
hwy2id          = ds["hwy2id"]
id2hwy          = ds["id2hwy"]
HIGHWAY_MASK_ID = ds["HIGHWAY_MASK_ID"]
scalers         = ds["scalers"]
num_highway     = ds["num_highway"]

from src.training.dataset import LANES_MASK_ID, ONEWAY_MASK_ID

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
from src.models.multi_attr_gat import MultiAttrGAT

model = MultiAttrGAT(num_highway=num_highway, cont_dim=12).to(device)

# ---------------------------------------------------------------------------
# Cell 2 (cont.) — Train
# ---------------------------------------------------------------------------
from src.training.trainer import train, plot_history, evaluate_test, save_checkpoint

history = train(
    model,
    data_train=data_train,
    data_val=data_val,
    data_test=data_test,
    num_highway=num_highway,
    highway_mask_id=HIGHWAY_MASK_ID,
    lanes_mask_id=LANES_MASK_ID,
    oneway_mask_id=ONEWAY_MASK_ID,
    device=device,
    epochs=EPOCHS,
    p_mask=P_MASK,
    eval_every=EVAL_EVERY,
)

plot_history(history)

# ---------------------------------------------------------------------------
# Cell 2 (cont.) — Test evaluation
# ---------------------------------------------------------------------------
from src.training.dataset import LANES_MISS_ID, ONEWAY_MISS_ID

evaluate_test(
    model,
    data_test=data_test,
    num_highway=num_highway,
    highway_mask_id=HIGHWAY_MASK_ID,
    lanes_mask_id=LANES_MASK_ID,
    oneway_mask_id=ONEWAY_MASK_ID,
    device=device,
    p_mask=P_MASK,
    test_seed=2025,
)

# ---------------------------------------------------------------------------
# Cell 3 — Save checkpoint
# ---------------------------------------------------------------------------
save_checkpoint(
    model,
    hwy2id=hwy2id,
    id2hwy=id2hwy,
    scalers=scalers,
    token_ids={
        "HIGHWAY_MASK_ID": HIGHWAY_MASK_ID,
        "LANES_MASK_ID":   LANES_MASK_ID,
        "LANES_MISS_ID":   LANES_MISS_ID,
        "ONEWAY_MASK_ID":  ONEWAY_MASK_ID,
        "ONEWAY_MISS_ID":  ONEWAY_MISS_ID,
    },
    meta={
        "seed":        SEED,
        "grid_size_m": 1000.0,
        "p_mask":      P_MASK,
        "city":        "jakarta",
    },
    save_path=str(CHECKPOINT_PATH),
)
