# %%
import sys
sys.path.append('/home/spatialuser/websites/mapedia')
from modules import DBHandler, DBUpdater, INTRA_CITY_LEARNING_SOURCE
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.MultiAttrGAT import MultiAttrGAT
from data.preprocessing import lanes_to_class, ZScaler, build_line_graph_edge_index
from torch_geometric.data import Data
from tqdm import tqdm
tqdm.pandas()

SEED = 42
EPOCHS = 2000
P_MASK = 0.30
MASK_TOKEN = "__MASK__"
UNK_TOKEN  = "__UNK__"
LANES_MASK_ID = 3   # lanes classes are 0,1,2; reserve 3 for MASK
LANES_MISS_ID = 4   # reserve 4 for MISSING (optional but helpful)
ONEWAY_MASK_VAL = -1  # sentinel in the categorical oneway ids (we'll use embedding-like ids)
ROAD_TYPE_MASK_ID = None

np.random.seed(SEED)
torch.manual_seed(SEED)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

db_handler = DBHandler()
df = db_handler.enriched_edges_from_bbox(
    miny= -6.3725962,
    maxy= -6.0785515,
    minx= 106.686105,
    maxx= 106.9737509
)


# Adjusting the values for Oneway column
df["oneway"] = df["oneway"].replace(3, 0)
df["oneway"] = df["oneway"].replace(2, 0)
df["oneway01"] = df["oneway"]
df["oneway01"] = df["oneway01"].fillna(0) # TODO: Needs to change


# Adjusting the road_type column
# road_type categorical vocab (add MASK and UNK)
road_type_vals = df["road_type"].astype(str).fillna("UNK").values
unique_road_types = sorted(pd.unique(road_type_vals).tolist())
# Ensure UNK exists (in case there were NaNs or unseen later)
if UNK_TOKEN not in unique_road_types:
    unique_road_types.append(UNK_TOKEN)
# Add MASK as special id
if MASK_TOKEN in unique_road_types:
    unique_road_types.remove(MASK_TOKEN)
unique_road_types.append(MASK_TOKEN)
road_type2id = {h:i for i,h in enumerate(unique_road_types)}
id2road_type = {i:h for h,i in road_type2id.items()}
ROAD_TYPE_MASK_ID = road_type2id[MASK_TOKEN]
df["road_type_id"] = df["road_type"].astype(str).map(lambda x: road_type2id.get(x, road_type2id[UNK_TOKEN])).astype(np.int64)


# Parse width to float
df["width_m"] = df["width"]


# Lanes class 0/1/2, missing -1
df["lanes_cls"] = df["nlanes"].apply(lanes_to_class).astype(np.int64)


df = df.reset_index(drop=True)
df["eid"] = np.arange(len(df), dtype=np.int64)


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
wid_scaler = ZScaler()
wid_scaler.fit(width_raw[train_idx])  # ignores NaN
width_z  = wid_scaler.transform(width_raw).astype(np.float32)  # NaN stays NaN
width_z_in  = np.nan_to_num(width_z, nan=0.0).astype(np.float32)

#%%
# =========================
# 6) Build PyG Data: base inputs + targets
# =========================
# We'll build inputs using:
# - continuous: length_z, width_z_in
# - categorical ids: road_type_id, lanes_input_id, oneway_input_id
#

# Edge index for line graph
edge_index = build_line_graph_edge_index(df, u_col="u", v_col="v", eid_col="eid")
print("Line graph edge_index:", edge_index.shape)

# %%
def build_TorchData(df):
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

    data = Data(
        # continuous base channels (1)
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
    return data.to(device)


data = build_TorchData(df)

# %%
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

# %%

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


    loss.backward()
    optimizer.step()

    # Eval
    val_metrics = evaluate(model, data, data.val_idx, p_mask=P_MASK)
    print(
        f"Epoch {epoch:02d} | loss={loss.item():.4f} "
        f"(hwy={loss_hwy.item():.3f}, lanes={loss_lan.item():.3f}, oneway={loss_onw.item():.3f}, "
        f"width={loss_wid.item():.3f} | "
        f"VAL masked: hwy_acc={val_metrics['hwy_acc_masked']:.3f}, "
        f"lan_acc={val_metrics['lan_acc_masked']:.3f}, "
        f"onw_acc={val_metrics['onw_acc_masked']:.3f}, "
        f"wid_MAE={val_metrics['wid_mae_masked']:.3f}, "
    )

# Final test metrics
test_metrics = evaluate(model, data, data.test_idx, p_mask=P_MASK)
print("TEST masked metrics:", test_metrics)
torch.save(model.state_dict(), "saved_models/gat_model.pth")


# %%





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
df.head()
df['road_type'] = pred['road_type']
df['lanes'] = pred['lanes']
df['oneway'] = pred['oneway']
df['width'] = pred['width']
df['road_type_conf'] = pred['road_type_conf']
df['lanes_conf'] = pred['lanes_conf']
df['oneway_conf'] = pred['oneway_conf']
df['width_conf'] = pred['width_conf']
output_df = df.dropna(subset=["road_id"])
df = output_df
df.isna().sum()
df['oneway'].isna().sum()
cols = ['road_type', 'lanes', 'oneway', 'width'] # Doesn't include maxspeed, minspeed
df_to_be_inserted = df[~df[cols].isna().any(axis=1)]
df_to_be_inserted.road_id = df_to_be_inserted.road_id.astype(int)

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
result = df_to_be_inserted.groupby(['road_id']).progress_apply(aggregate_predictions)
result.shape
result.index.name = None
result
result.width.value_counts()
result['lanes_source'] = INTRA_CITY_LEARNING_SOURCE
result['oneway_source'] = INTRA_CITY_LEARNING_SOURCE
result['road_type_source'] = INTRA_CITY_LEARNING_SOURCE
result['width_source'] = INTRA_CITY_LEARNING_SOURCE
result.rename(columns={
    'lanes': 'nlanes',
    'lanes_conf': 'nlanes_conf',
    'lanes_source': 'nlanes_source'
}, inplace=True)
result.oneway = result.oneway.map({False: 0, True: 1})
result
#from modules import DBUpdater
db_updater = DBUpdater(db_handler)
db_updater.update_database(static_attr=result, static_cols=['nlanes', 'width', 'oneway', 'road_type'])
db_updater.road_attributes
result

# %%
