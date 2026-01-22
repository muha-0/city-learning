import sys
sys.path.append('/home/spatialuser/websites/mapedia')
# Replica 1
from modules import DBHandler

db_handler = DBHandler()
db_handler.connect_to_db()
G = db_handler.get_graph(
    min_lat= -6.3725962,
    max_lat= -6.0785515,
    min_lon= 106.686105,
    max_lon= 106.9737509
)
type(G)
import matplotlib.pyplot as plt
import osmnx as ox
print("Plotting graph...")
fig, ax = ox.plot_graph(G, show=False, close=False,  node_size=5, edge_color='gray')
plt.show()
# Get the first node's ID and its attributes
first_node_data = list(G.nodes(data=True))[0]
print(first_node_data)
# Get the first edge's data (start_node, end_node, key, attributes)
first_edge_data = list(G.edges(data=True, keys=True))[4676]
print(first_edge_data)
import osmnx as ox
import pandas as pd
import numpy as np 

gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

gdf_edges = gdf_edges.reset_index()

print("--- Edge DataFrame Info (shows non-null counts) ---")
gdf_edges.info()

# Get an exact count of missing values for each column
print("\n--- Count of Missing Values per Column ---")
print(gdf_edges.isnull().sum())
gdf_edges
print(gdf_edges["maxspeed"].value_counts())
# gdf_edges.rename(columns={"osm_id": "osmid"}, inplace=True)
import psycopg2
from sqlalchemy import create_engine, text
import sys

# --- Configuration ---
DB_HOST = 'cs-u-spatial-406.cs.umn.edu'
DB_NAME = 'gis'
DB_USER = 'gis'
DB_PASS = 'gis'
DB_PORT = 5432

# Setup the connection 
connection_str = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(connection_str)

with engine.connect() as connection:
    
    result = connection.execute(text("SELECT * FROM road_attributes LIMIT 0"))
    columns = result.keys()
    
    print("--- Columns in 'road_attributes' table ---")
    for col in columns:
        print(col)
import psycopg2
from sqlalchemy import create_engine, text
import sys

# --- Configuration ---
DB_HOST = 'cs-u-spatial-406.cs.umn.edu'
DB_NAME = 'gis'
DB_USER = 'gis'
DB_PASS = 'gis'
DB_PORT = 5432


connection_str = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(connection_str)

def simplify_id(val):
    if isinstance(val, list):
        return val[0] # Take the first ID if it's a list
    return val

# Apply it to the osmid column
gdf_edges['osmid'] = gdf_edges['osmid'].apply(simplify_id)

# Get the list of OSM IDs from local DataFrame
# Ensure they are integers
ids_to_fetch = gdf_edges['osmid'].astype(int).unique().tolist()

print(f"Looking up {len(ids_to_fetch)} unique OSM IDs in the remote database...")

# Define a function to fetch data in chunks to avoid crashing the query
def fetch_attributes_in_chunks(id_list, chunk_size=1_000_000):
    all_results = []
    
    # Loop through the IDs in steps of 'chunk_size'
    for i in range(0, len(id_list), chunk_size):
        chunk = id_list[i : i + chunk_size]
        
        # Format the IDs for SQL (e.g., "123, 456, 789")
        ids_tuple = str(tuple(chunk)).replace(',)', ')') # Handle single-element tuple quirk
        
        # Query: Select only the columns we need
        query = f"""
            SELECT osm_id, nlanes, width, max_speed, min_speed, oneway, road_type
            FROM road_attributes 
            WHERE osm_id IN {ids_tuple}
        """
        
        # Run query and append to list
        chunk_df = pd.read_sql(query, engine)
        all_results.append(chunk_df)
        
        print(f"  Fetched batch {i} to {i+chunk_size}...")

    # Combine all chunks into one DataFrame
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

# Execute the fetch
db_data = fetch_attributes_in_chunks(ids_to_fetch)

# Clean up the external data before merging
# Rename columns to match your 'osmnx' DataFrame for easier merging later
db_data = db_data.rename(columns={
    'osm_id': 'osmid',
    'nlanes': 'lanes_db',
    'width': 'width_db',
    'max_speed': 'maxspeed_db',
    'min_speed': 'minspeed_db',
    'oneway': 'oneway_db',
    'road_type': 'road_type_db'
})

# Ensure osmid is the same type (string/object) as your local df for merging
db_data['osmid'] = db_data['osmid'].astype(str)

print(f"\nSuccessfully retrieved {len(db_data)} rows from the database.")
print(db_data.head())
db_data
print(db_data.info())

print(gdf_edges.info())
# gdf_edges['width'] = gdf_edges['width'].fillna(gdf_edges['est_width']) Will do it lastly
# columns_to_drop = ["junction", "bridge", "access", "ref","tunnel", "reversed", "name"]
# gdf_edges = gdf_edges.drop(columns = columns_to_drop)
# print(gdf_edges.isnull().sum())
# print(gdf_edges["maxspeed"].value_counts())
# print(gdf_edges["lanes"].value_counts())
# for val in gdf_edges["width"].value_counts().index:
#     print(val)
# print(gdf_edges["road_type"].value_counts())
gdf_edges
db_data.osmid = db_data.osmid.astype(int)
merged_df = gdf_edges.merge(db_data, on='osmid', how='left')
merged_df.drop(columns=[
    'key',
    'oneway',
    'length',
    'maxspeed'
    ], inplace=True)
merged_df.rename(columns={
    'lanes_db': 'lanes', 
    'width_db': 'width',
    'maxspeed_db': 'maxspeed',
    'minspeed_db': 'minspeed',
    'oneway_db': 'oneway',
    'road_type_db': 'road_type'
    }, inplace=True)
merged_df.head()
merged_df.oneway = merged_df.oneway.astype(bool)
merged_df["maxspeed"] = (
    pd.to_numeric(merged_df["maxspeed"], errors="coerce")
      .astype("Int64")   # capital I
)
merged_df["minspeed"] = (
    pd.to_numeric(merged_df["minspeed"], errors="coerce")
      .astype("Int64")   # capital I
)
merged_df["lanes"] = (
    pd.to_numeric(merged_df["lanes"], errors="coerce")
      .astype("Int64")   # capital I
)
merged_df["road_type"] = (
    pd.to_numeric(merged_df["road_type"], errors="coerce")
      .astype("Int64")   # capital I
)
merged_df.info()



print(merged_df.isnull().sum())
# 5. Cleanup
# Drop the temporary columns we just used
final_gdf = merged_df #.drop(columns=['maxspeed', 'minspeed'])

# 6. Check the Results
print("\n--- Missing Values BEFORE Database Fill ---")
print(merged_df.isnull().sum())

print("\n--- Missing Values AFTER Database Fill ---")
print(final_gdf.isnull().sum())
print(final_gdf.info())
# final_gdf['width'] = final_gdf['width'].fillna(final_gdf['est_width'])
# final_gdf = final_gdf.drop(columns = ["est_width", "maxspeed"])
print(final_gdf.info())

# =========================
# 0) Imports
# =========================
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# =========================
# 1) Helpers: parsing & encoding
# =========================

def lanes_to_class(x):
    """
    x: numeric lanes value or NaN
    returns:
      -1 if NaN
       0 for 1 lane
       1 for 2 lanes
       2 for 3+ lanes
    """
    if pd.isna(x):
        return -1
    # If lanes are floats like 1.0, 2.0, 3.0
    v = float(x)

    if v <= 1.0:
        return 0
    elif v <= 2.0:
        return 1
    else:
        return 2

class ZScaler:
    """Z-score scaler that ignores NaN."""
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, x: np.ndarray):
        self.mu = np.nanmean(x)
        self.sd = np.nanstd(x) + 1e-8

    def transform(self, x: np.ndarray):
        return (x - self.mu) / self.sd


# =========================
# 2) Build line graph adjacency from (u,v)
# =========================
def build_line_graph_edge_index(df, u_col="u", v_col="v", eid_col="eid"):
    """
    Nodes in line graph = edges in original graph (df rows).
    Two line-graph nodes connect if original edges share an endpoint (u or v).
    """
    incident = {}
    u_vals = df[u_col].values
    v_vals = df[v_col].values
    eids  = df[eid_col].values

    for u, v, eid in zip(u_vals, v_vals, eids):
        incident.setdefault(u, []).append(eid)
        incident.setdefault(v, []).append(eid)

    pairs = []
    for node, lst in incident.items():
        m = len(lst)
        if m < 2:
            continue
        # clique among incident edges
        for i in range(m):
            for j in range(i + 1, m):
                a, b = lst[i], lst[j]
                pairs.append((a, b))
                pairs.append((b, a))

    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return edge_index


# =========================
# 3) Prepare dataframe: ids, parsing, vocab
# =========================
df = final_gdf.copy()

# Stable node id for line graph nodes
df = df.reset_index(drop=True)
df["eid"] = np.arange(len(df), dtype=np.int64)

# Parse width to float
df["width_m"] = df["width"]

# Lanes class 0/1/2, missing -1
df["lanes_cls"] = df["lanes"].apply(lanes_to_class).astype(np.int64)

# Oneway to 0/1 int
# your final_gdf shows oneway is int64 already; this keeps it safe:
df["oneway01"] = df["oneway"].astype(int).astype(np.int64)

# road_type categorical vocab (add MASK and UNK)
road_type_vals = df["road_type"].astype(str).fillna("UNK").values
unique_road_types = sorted(pd.unique(road_type_vals).tolist())

MASK_TOKEN = "__MASK__"
UNK_TOKEN  = "__UNK__"

# Ensure UNK exists (in case there were NaNs or unseen later)
if UNK_TOKEN not in unique_road_types:
    unique_road_types.append(UNK_TOKEN)
# Add MASK as special id
if MASK_TOKEN in unique_road_types:
    unique_road_types.remove(MASK_TOKEN)
unique_road_types.append(MASK_TOKEN)
road_type2id = {h:i for i,h in enumerate(unique_road_types)}
id2road_type = {i:h for h,i in road_type2id.items()}

df["road_type_id"] = df["road_type"].astype(str).map(lambda x: road_type2id.get(x, road_type2id[UNK_TOKEN])).astype(np.int64)

# Special IDs
ROAD_TYPE_MASK_ID = road_type2id[MASK_TOKEN]

LANES_MASK_ID = 3   # lanes classes are 0,1,2; reserve 3 for MASK
LANES_MISS_ID = 4   # reserve 4 for MISSING (optional but helpful)

ONEWAY_MASK_VAL = -1  # sentinel in the categorical oneway ids (we'll use embedding-like ids)


# =========================
# 4) Train/Val/Test split (node-level)
# =========================
N = len(df)
perm = np.random.permutation(N)
n_train = int(0.80 * N)
n_val   = int(0.10 * N)

train_idx = perm[:n_train]
val_idx   = perm[n_train:n_train+n_val]
test_idx  = perm[n_train+n_val:]

train_mask = np.zeros(N, dtype=bool); train_mask[train_idx] = True
val_mask   = np.zeros(N, dtype=bool); val_mask[val_idx] = True
test_mask  = np.zeros(N, dtype=bool); test_mask[test_idx] = True


# =========================
# 5) Normalize continuous features (train-only, ignore NaNs)
# =========================
# # Length: log1p then z-score (length appears complete, but we do it generally)
# length_raw = df["length"].to_numpy(dtype=np.float32)
# length_log = np.log1p(length_raw)

width_raw = df["width_m"].to_numpy(dtype=np.float32)  # contains NaN

len_scaler = ZScaler()
wid_scaler = ZScaler()

# len_scaler.fit(length_log[train_idx])
wid_scaler.fit(width_raw[train_idx])  # ignores NaN

# length_z = len_scaler.transform(length_log).astype(np.float32)
width_z  = wid_scaler.transform(width_raw).astype(np.float32)  # NaN stays NaN

# For INPUTS: replace NaN with 0.0 (mask token for continuous)
width_z_in  = np.nan_to_num(width_z, nan=0.0).astype(np.float32)


# =========================
# 6) Build PyG Data: base inputs + targets
# =========================
# We'll build inputs using:
# - continuous: length_z, width_z_in
# - categorical ids: road_type_id, lanes_input_id, oneway_input_id
#
# For lanes input id: use lanes_cls when present else LANES_MISS_ID
lanes_in_id = df["lanes_cls"].to_numpy(dtype=np.int64)
lanes_in_id = np.where(lanes_in_id == -1, LANES_MISS_ID, lanes_in_id)

# For oneway input id: map 0/1 to 0/1, no missing in your data, but reserve MASK later
oneway_in_id = df["oneway01"].to_numpy(dtype=np.int64)  # 0/1

# Targets
y_road_type = df["road_type_id"].to_numpy(dtype=np.int64)
y_lanes   = df["lanes_cls"].to_numpy(dtype=np.int64)    # -1 for missing
y_oneway  = df["oneway01"].to_numpy(dtype=np.float32)   # BCE expects float 0/1
y_width   = df["width_m"].to_numpy(dtype=np.float32)    # NaN allowed
# y_length  = length_log.astype(np.float32)               # use log1p length as regression target

# Edge index for line graph
edge_index = build_line_graph_edge_index(df, u_col="u", v_col="v", eid_col="eid")
print("Line graph edge_index:", edge_index.shape)

data = Data(
    # continuous base channels (2)
    # x_cont=torch.tensor(np.column_stack([length_z, width_z_in]), dtype=torch.float32),
    x_cont=torch.tensor(np.column_stack([width_z_in]), dtype=torch.float32),
    edge_index=edge_index
)

# Store categorical input ids
data.road_type_in = torch.tensor(df["road_type_id"].to_numpy(dtype=np.int64), dtype=torch.long)
data.lanes_in   = torch.tensor(lanes_in_id, dtype=torch.long)
data.oneway_in  = torch.tensor(oneway_in_id, dtype=torch.long)

# Store targets
data.y_road_type = torch.tensor(y_road_type, dtype=torch.long)
data.y_lanes   = torch.tensor(y_lanes, dtype=torch.long)
data.y_oneway  = torch.tensor(y_oneway, dtype=torch.float32)
data.y_width   = torch.tensor(y_width, dtype=torch.float32)
# data.y_length  = torch.tensor(y_length, dtype=torch.float32)

# Splits
data.train_idx = torch.tensor(train_idx, dtype=torch.long)
data.val_idx   = torch.tensor(val_idx, dtype=torch.long)
data.test_idx  = torch.tensor(test_idx, dtype=torch.long)

# Move static tensors to device later; we will clone inputs per epoch for masking
data = data.to(device)

# =========================
# 7) Model: embeddings + 2-layer GATv2Conv + multi-head decoders
# =========================
# from src.models.MultiAttrGAT import MultiAttrGAT

num_road_type = len(road_type2id)
model = MultiAttrGAT(num_road_type=num_road_type).to(device)

# Losses
loss_ce = nn.CrossEntropyLoss()
loss_bce = nn.BCEWithLogitsLoss()
loss_huber = nn.SmoothL1Loss()


# =========================
# 8) Masking utilities (30% per attribute, only on valid, only train nodes)
# =========================
def bernoulli_mask(idx, valid_mask, p=0.3):
    """
    idx: LongTensor of train node indices
    valid_mask: BoolTensor [N] validity
    returns BoolTensor [N] masking positions
    """
    m = torch.zeros_like(valid_mask, dtype=torch.bool)
    # sample only on idx
    r = torch.rand(idx.shape[0], device=idx.device)
    chosen = r < p
    m[idx[chosen]] = True
    # intersect with validity
    m = m & valid_mask
    return m

# Special ids for masking categorical inputs
ONEWAY_MASK_ID = 2  # in the oneway embedding (0,1, MASK=2)

# continuous input column indices
# CONT_LENGTH_COL = 0
CONT_WIDTH_COL  = 1
CONT_WIDTH_COL  = 0 # since we have only width now


# =========================
# 9) Training loop
# =========================
def evaluate(model, data, split_idx, p_mask=0.3):
    """
    Evaluate masked-only reconstruction on a split.
    For evaluation, we create fresh masks on the split and measure metrics on masked positions.
    """
    model.eval()
    with torch.no_grad():
        idx = split_idx

        # Validity masks
        valid_hwy = torch.ones_like(data.y_road_type, dtype=torch.bool)
        valid_lan = (data.y_lanes != -1)
        valid_onw = torch.ones_like(data.y_oneway, dtype=torch.bool)
        valid_wid = ~torch.isnan(data.y_width)
        # valid_len = ~torch.isnan(data.y_length)

        # Create masking positions on this split
        mask_hwy = bernoulli_mask(idx, valid_hwy, p_mask)
        mask_lan = bernoulli_mask(idx, valid_lan, p_mask)
        mask_onw = bernoulli_mask(idx, valid_onw, p_mask)
        mask_wid = bernoulli_mask(idx, valid_wid, p_mask)
        # mask_len = bernoulli_mask(idx, valid_len, p_mask)

        # Corrupt inputs
        x_cont = data.x_cont.clone()
        road_type_in = data.road_type_in.clone()
        lanes_in = data.lanes_in.clone()
        oneway_in = data.oneway_in.clone()

        # categorical mask tokens
        road_type_in[mask_hwy] = ROAD_TYPE_MASK_ID
        lanes_in[mask_lan]   = LANES_MASK_ID
        oneway_in[mask_onw]  = ONEWAY_MASK_ID

        # continuous mask tokens
        x_cont[mask_wid, CONT_WIDTH_COL]  = 0.0
        # x_cont[mask_len, CONT_LENGTH_COL] = 0.0

        pred = model(x_cont, road_type_in, lanes_in, oneway_in, data.edge_index)

        # Metrics (masked-only)
        # road_type accuracy
        hwy_acc = float((pred["road_type"][mask_hwy].argmax(dim=1) == data.y_road_type[mask_hwy]).float().mean()) if mask_hwy.any() else np.nan
        lan_acc = float((pred["lanes"][mask_lan].argmax(dim=1) == data.y_lanes[mask_lan]).float().mean()) if mask_lan.any() else np.nan

        # oneway accuracy
        if mask_onw.any():
            onw_prob = torch.sigmoid(pred["oneway"][mask_onw])
            onw_hat = (onw_prob >= 0.5).float()
            onw_acc = float((onw_hat == data.y_oneway[mask_onw]).float().mean())
        else:
            onw_acc = np.nan

        # width MAE
        wid_mae = float(torch.mean(torch.abs(pred["width"][mask_wid] - data.y_width[mask_wid]))) if mask_wid.any() else np.nan
        # len_mae = float(torch.mean(torch.abs(pred["length"][mask_len] - data.y_length[mask_len]))) if mask_len.any() else np.nan

        return {
            "hwy_acc_masked": hwy_acc,
            "lan_acc_masked": lan_acc,
            "onw_acc_masked": onw_acc,
            "wid_mae_masked": wid_mae,
            # "len_mae_masked": len_mae,
        }

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCHS = 2000
P_MASK = 0.30

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    idx = data.train_idx

    # Validity masks
    valid_hwy = torch.ones_like(data.y_road_type, dtype=torch.bool)
    valid_lan = (data.y_lanes != -1)
    valid_onw = torch.ones_like(data.y_oneway, dtype=torch.bool)
    valid_wid = ~torch.isnan(data.y_width)
    # valid_len = ~torch.isnan(data.y_length)

    # Create masking positions (train only)
    mask_hwy = bernoulli_mask(idx, valid_hwy, P_MASK)
    mask_lan = bernoulli_mask(idx, valid_lan, P_MASK)
    mask_onw = bernoulli_mask(idx, valid_onw, P_MASK)
    mask_wid = bernoulli_mask(idx, valid_wid, P_MASK)
    # mask_len = bernoulli_mask(idx, valid_len, P_MASK)

    # Corrupt inputs
    x_cont = data.x_cont.clone()
    road_type_in = data.road_type_in.clone()
    lanes_in = data.lanes_in.clone()
    oneway_in = data.oneway_in.clone()

    # categorical masks
    road_type_in[mask_hwy] = ROAD_TYPE_MASK_ID
    lanes_in[mask_lan]   = LANES_MASK_ID
    oneway_in[mask_onw]  = ONEWAY_MASK_ID

    # continuous masks
    x_cont[mask_wid, CONT_WIDTH_COL]  = 0.0
    # x_cont[mask_len, CONT_LENGTH_COL] = 0.0

    pred = model(x_cont, road_type_in, lanes_in, oneway_in, data.edge_index)

    # Compute losses only on masked positions
    loss = 0.0

    # road_type CE
    if mask_hwy.any():
        loss_hwy = loss_ce(pred["road_type"][mask_hwy], data.y_road_type[mask_hwy])
        loss = loss + loss_hwy
    else:
        loss_hwy = torch.tensor(0.0, device=device)

    # lanes CE (only where lanes exist)
    if mask_lan.any():
        loss_lan = loss_ce(pred["lanes"][mask_lan], data.y_lanes[mask_lan])
        loss = loss + loss_lan
    else:
        loss_lan = torch.tensor(0.0, device=device)

    # oneway BCE
    if mask_onw.any():
        loss_onw = loss_bce(pred["oneway"][mask_onw], data.y_oneway[mask_onw])
        loss = loss + loss_onw
    else:
        loss_onw = torch.tensor(0.0, device=device)

    # width Huber (only where width exists)
    if mask_wid.any():
        loss_wid = loss_huber(pred["width"][mask_wid], data.y_width[mask_wid])
        loss = loss + loss_wid
    else:
        loss_wid = torch.tensor(0.0, device=device)

    # # length Huber
    # if mask_len.any():
    #     loss_len = loss_huber(pred["length"][mask_len], data.y_length[mask_len])
    #     loss = loss + loss_len
    # else:
    #     loss_len = torch.tensor(0.0, device=device)

    loss.backward()
    optimizer.step()

    # Eval
    val_metrics = evaluate(model, data, data.val_idx, p_mask=P_MASK)
    print(
        f"Epoch {epoch:02d} | loss={loss.item():.4f} "
        f"(hwy={loss_hwy.item():.3f}, lanes={loss_lan.item():.3f}, oneway={loss_onw.item():.3f}, "
        # f"width={loss_wid.item():.3f}, length={loss_len.item():.3f}) | "
        f"width={loss_wid.item():.3f} | "
        f"VAL masked: hwy_acc={val_metrics['hwy_acc_masked']:.3f}, "
        f"lan_acc={val_metrics['lan_acc_masked']:.3f}, "
        f"onw_acc={val_metrics['onw_acc_masked']:.3f}, "
        f"wid_MAE={val_metrics['wid_mae_masked']:.3f}, "
        # f"len_MAE={val_metrics['len_mae_masked']:.3f}"
    )

# Final test metrics
test_metrics = evaluate(model, data, data.test_idx, p_mask=P_MASK)
print("TEST masked metrics:", test_metrics)
device
torch.save(model.state_dict(), "saved_models/gat_model.pth")
model = MultiAttrGAT(num_road_type=num_road_type).to(device)
model.load_state_dict(torch.load("saved_models/gat_model.pth", map_location="cpu"))
model.eval()  # important for inference
# Final test metrics
test_metrics = evaluate(model, data, data.test_idx, p_mask=P_MASK)
print("TEST masked metrics:", test_metrics)
# How to upload the data back to database 

model
data
data.edge_index
def predict(model, data, split_idx, p_mask=0.3):
    """
    Evaluate masked-only reconstruction on a split.
    For evaluation, we create fresh masks on the split and measure metrics on masked positions.
    """
    model.eval()
    with torch.no_grad():
        idx = split_idx

        # Validity masks
        valid_hwy = torch.ones_like(data.y_road_type, dtype=torch.bool)
        valid_lan = (data.y_lanes != -1)
        valid_onw = torch.ones_like(data.y_oneway, dtype=torch.bool)
        valid_wid = ~torch.isnan(data.y_width)
        # valid_len = ~torch.isnan(data.y_length)

        # Create masking positions on this split
        mask_hwy = bernoulli_mask(idx, valid_hwy, p_mask)
        mask_lan = bernoulli_mask(idx, valid_lan, p_mask)
        mask_onw = bernoulli_mask(idx, valid_onw, p_mask)
        mask_wid = bernoulli_mask(idx, valid_wid, p_mask)
        # mask_len = bernoulli_mask(idx, valid_len, p_mask)

        # Corrupt inputs
        x_cont = data.x_cont.clone()
        road_type_in = data.road_type_in.clone()
        lanes_in = data.lanes_in.clone()
        oneway_in = data.oneway_in.clone()

        # categorical mask tokens
        road_type_in[mask_hwy] = ROAD_TYPE_MASK_ID
        lanes_in[mask_lan]   = LANES_MASK_ID
        oneway_in[mask_onw]  = ONEWAY_MASK_ID

        # continuous mask tokens
        x_cont[mask_wid, CONT_WIDTH_COL]  = 0.0
        # x_cont[mask_len, CONT_LENGTH_COL] = 0.0

        pred = model(x_cont, road_type_in, lanes_in, oneway_in, data.edge_index)

        conf, val = torch.max(pred["road_type"], dim=1)
        pred["road_type"] = val.cpu().numpy()
        pred["road_type_conf"] = conf.cpu().numpy()

        conf, val = torch.max(pred["lanes"], dim=1)
        pred["lanes"] = val.cpu().numpy()
        pred["lanes_conf"] = conf.cpu().numpy()

        pred["oneway_conf"] = pred["oneway"].cpu().numpy()
        pred["oneway"] = (pred["oneway_conf"] >= 0.5)

        pred["width"] = pred["width"].cpu().numpy()

        sigma = pred["width"].std(axis=0)

        pred["width_conf"] = np.full(pred["width"].shape, 1 / (1 + sigma))

        return pred
pred = predict(model, data, data.test_idx, p_mask=0.3)
pred['road_type'], pred['road_type'].shape
pred['lanes'], pred['lanes'].shape
pred['oneway'], pred['oneway'].shape
pred['width'], pred['width'].shape
pred
df['road_type'] = pred['road_type']
df['lanes'] = pred['lanes']
df['oneway'] = pred['oneway']
df['width'] = pred['width']
df['road_type_conf'] = pred['road_type_conf']
df['lanes_conf'] = pred['lanes_conf']
df['oneway_conf'] = pred['oneway_conf']
df['width_conf'] = pred['width_conf']
df
df.isna().sum()
df['oneway'].isna().sum()
cols = ['road_type', 'lanes', 'oneway', 'width'] # Doesn't include maxspeed, minspeed
df_to_be_inserted = df[~df[cols].isna().any(axis=1)]
df_to_be_inserted
from tqdm import tqdm
tqdm.pandas()
def aggregate_predictions(group):
    result = {}
    for col in ['osmid', 'road_type', 'lanes', 'oneway', 'width', 'road_type_conf', 'lanes_conf', 'oneway_conf', 'width_conf']:
        if col in group:
            if col in ['width_conf', 'width']:
                # For continuous, take mean
                val = group[col].mean()
                result[col] = pd.NA if pd.isna(val) else val
            else:
                # For categorical, take mode
                mode = group[col].mode(dropna=True)
                result[col] = mode.iloc[0] if not mode.empty else pd.NA
    return pd.Series(result)
result = df_to_be_inserted.groupby(['id']).progress_apply(aggregate_predictions)
result.shape
result.index.name = None
result
result.width.value_counts()
TECHNIQUE_SOURCE = 4
result['lanes_source'] = TECHNIQUE_SOURCE
result['oneway_source'] = TECHNIQUE_SOURCE
result['road_type_source'] = TECHNIQUE_SOURCE
result['width_source'] = TECHNIQUE_SOURCE
result.rename(columns={
    'lanes': 'nlanes',
    'lanes_conf': 'nlanes_conf',
    'lanes_source': 'nlanes_source'
}, inplace=True)
from modules import DBUpdater
db_updater = DBUpdater(db_handler)
db_updater.update_database(static_attr=result, static_cols=['nlanes', 'width', 'oneway', 'road_type'])
