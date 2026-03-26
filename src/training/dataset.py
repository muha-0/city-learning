"""
Build PyTorch Geometric Data objects from a road-edge GeoDataFrame.

Covers:
  - Line-graph construction  (edges of original graph become nodes)
  - Vocab encoding for categorical features
  - Spatial-inductive BFS train/val/test split
  - Z-score normalisation of continuous features
  - Induced subgraph construction per split
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch_geometric.data import Data

from ..utils.utils import ZScaler, lanes_to_class


# ---------------------------------------------------------------------------
# Constants for special token IDs
# ---------------------------------------------------------------------------
MASK_TOKEN = "__MASK__"
UNK_TOKEN = "__UNK__"

LANES_MASK_ID = 3   # classes: 0,1,2 | MASK=3 | MISSING=4
LANES_MISS_ID = 4

ONEWAY_MASK_ID = 2  # values: 0,1 | MASK=2 | MISSING=3
ONEWAY_MISS_ID = 3


# ---------------------------------------------------------------------------
# Line-graph construction
# ---------------------------------------------------------------------------

def build_line_graph_edge_index(
    df: pd.DataFrame,
    u_col: str = "u",
    v_col: str = "v",
    eid_col: str = "eid",
) -> torch.Tensor:
    """
    Build a line-graph edge index from an edge GeoDataFrame.

    In the line graph:
      - Nodes  = rows (edges) of the original graph.
      - Edges  = pairs of original edges sharing an endpoint.

    Returns a [2, E] int64 tensor.
    """
    incident: dict[int, list[int]] = {}
    for u, v, eid in zip(df[u_col].values, df[v_col].values, df[eid_col].values):
        incident.setdefault(int(u), []).append(int(eid))
        incident.setdefault(int(v), []).append(int(eid))

    pairs: list[tuple[int, int]] = []
    for lst in incident.values():
        m = len(lst)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                a, b = lst[i], lst[j]
                pairs.append((a, b))
                pairs.append((b, a))

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


# ---------------------------------------------------------------------------
# Vocab building
# ---------------------------------------------------------------------------

def build_highway_vocab(highway_series: pd.Series) -> tuple[dict, dict, int]:
    """Return hwy2id, id2hwy, HIGHWAY_MASK_ID."""
    vals = highway_series.fillna(UNK_TOKEN).astype(str).values
    unique = sorted(pd.unique(vals).tolist())
    if UNK_TOKEN not in unique:
        unique.append(UNK_TOKEN)
    if MASK_TOKEN in unique:
        unique.remove(MASK_TOKEN)
    unique.append(MASK_TOKEN)
    hwy2id = {h: i for i, h in enumerate(unique)}
    id2hwy = {i: h for h, i in hwy2id.items()}
    return hwy2id, id2hwy, hwy2id[MASK_TOKEN]


# ---------------------------------------------------------------------------
# Spatial-inductive BFS split
# ---------------------------------------------------------------------------

def _bfs_grow_region(
    target_edges: int,
    available_cells: set,
    seed_cell: int,
    rng: np.random.Generator,
    cell_edge_count: dict,
    neighbors: dict,
) -> tuple[set, int]:
    region: set[int] = set()
    total = 0
    q = [seed_cell]
    in_queue = {seed_cell}

    while q and total < target_edges:
        cur = q.pop(0)
        in_queue.discard(cur)
        if cur not in available_cells or cur in region:
            continue
        region.add(cur)
        total += cell_edge_count.get(cur, 0)

        nbrs = neighbors.get(cur, [])
        if len(nbrs) > 1:
            rng.shuffle(nbrs)
        for nb in nbrs:
            if nb in available_cells and nb not in region and nb not in in_queue:
                q.append(nb)
                in_queue.add(nb)

        if not q and total < target_edges:
            remaining = list(available_cells - region)
            if not remaining:
                break
            new_seed = int(rng.choice(remaining))
            q.append(new_seed)
            in_queue.add(new_seed)

    return region, total


def spatial_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    grid_size_m: float = 1000.0,
    utm_epsg: int = 32748,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contiguous BFS-based spatial-inductive split.

    Parameters
    ----------
    df : GeoDataFrame with 'cx', 'cy' columns (projected metres).
    Returns train_idx, val_idx, test_idx as sorted int64 arrays.
    """
    N = len(df)
    gx = np.floor(df["cx"].to_numpy() / grid_size_m).astype(np.int64)
    gy = np.floor(df["cy"].to_numpy() / grid_size_m).astype(np.int64)

    cell_pairs = list(zip(gx.tolist(), gy.tolist()))
    cell_id = pd.factorize(pd.Series(cell_pairs))[0].astype(np.int64)
    df = df.copy()
    df["cell_id"] = cell_id

    unique_cells = np.unique(cell_id)

    cell_to_coord: dict[int, tuple] = {}
    coord_to_cell: dict[tuple, int] = {}
    for cid, pair in zip(df["cell_id"].to_numpy(), cell_pairs):
        if int(cid) not in cell_to_coord:
            cell_to_coord[int(cid)] = (int(pair[0]), int(pair[1]))
            coord_to_cell[(int(pair[0]), int(pair[1]))] = int(cid)

    cell_to_edges: dict[int, list] = {}
    for idx, cid in enumerate(df["cell_id"].to_numpy()):
        cell_to_edges.setdefault(int(cid), []).append(int(idx))

    cell_edge_count = {cid: len(idxs) for cid, idxs in cell_to_edges.items()}

    neighbors: dict[int, list] = {int(cid): [] for cid in unique_cells.tolist()}
    for cid in unique_cells.tolist():
        x, y = cell_to_coord[int(cid)]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb_coord = (x + dx, y + dy)
            if nb_coord in coord_to_cell:
                neighbors[int(cid)].append(coord_to_cell[nb_coord])

    rng = np.random.default_rng(seed)
    available = set(unique_cells.tolist())

    target_test = int(test_frac * N)
    target_val = int(val_frac * N)

    cells_sorted = sorted(list(available), key=lambda c: cell_edge_count.get(c, 0), reverse=True)
    topk = max(1, int(0.2 * len(cells_sorted)))
    seed_test = int(rng.choice(cells_sorted[:topk])) if cells_sorted else int(rng.choice(list(available)))

    test_cells, _ = _bfs_grow_region(target_test, available, seed_test, rng, cell_edge_count, neighbors)
    available -= test_cells

    boundary = {nb for c in test_cells for nb in neighbors.get(c, []) if nb in available}
    if boundary:
        seed_val = int(rng.choice(list(boundary)))
    else:
        cells_sorted2 = sorted(list(available), key=lambda c: cell_edge_count.get(c, 0), reverse=True)
        topk2 = max(1, int(0.2 * len(cells_sorted2)))
        seed_val = int(rng.choice(cells_sorted2[:topk2])) if cells_sorted2 else int(rng.choice(list(available)))

    val_cells, _ = _bfs_grow_region(target_val, available, seed_val, rng, cell_edge_count, neighbors)
    available -= val_cells
    train_cells = available

    def collect(cells):
        idxs = np.array([i for c in cells for i in cell_to_edges.get(c, [])], dtype=np.int64)
        idxs.sort()
        return idxs

    train_idx = collect(train_cells)
    val_idx = collect(val_cells)
    test_idx = collect(test_cells)

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Feature normalisation
# ---------------------------------------------------------------------------

def normalize_features(
    df: pd.DataFrame,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, ZScaler]]:
    """
    Fit Z-score scalers on training data and return the full [N, 12] x_cont array
    plus the fitted scalers.

    x_cont columns:
      0  length_z   1  width_z    2  max_z     3  min_z
      4  len_miss   5  wid_miss   6  max_miss  7  min_miss
      8  len_mask   9  wid_mask  10  max_mask  11 min_mask
    """
    length_log = np.log1p(df["length"].to_numpy(dtype=np.float32))
    width_raw = df["width_m"].to_numpy(dtype=np.float32)
    max_raw = df["max_speed_val"].to_numpy(dtype=np.float32)
    min_raw = df["min_speed_val"].to_numpy(dtype=np.float32)

    scalers: dict[str, ZScaler] = {}
    for name, arr in [("length", length_log), ("width", width_raw), ("max", max_raw), ("min", min_raw)]:
        s = ZScaler()
        s.fit(arr[train_idx])
        scalers[name] = s

    length_z = scalers["length"].transform(length_log).astype(np.float32)
    width_z = scalers["width"].transform(width_raw).astype(np.float32)
    max_z = scalers["max"].transform(max_raw).astype(np.float32)
    min_z = scalers["min"].transform(min_raw).astype(np.float32)

    width_missing = np.isnan(width_z).astype(np.float32)
    length_missing = np.zeros_like(width_missing)
    max_missing = np.isnan(max_z).astype(np.float32)
    min_missing = np.isnan(min_z).astype(np.float32)

    width_z = np.nan_to_num(width_z, nan=0.0)
    max_z = np.nan_to_num(max_z, nan=0.0)
    min_z = np.nan_to_num(min_z, nan=0.0)

    N = len(df)
    x_cont_all = np.column_stack([
        length_z, width_z, max_z, min_z,
        length_missing, width_missing, max_missing, min_missing,
        np.zeros(N, dtype=np.float32),  # len_mask
        np.zeros(N, dtype=np.float32),  # wid_mask
        np.zeros(N, dtype=np.float32),  # max_mask
        np.zeros(N, dtype=np.float32),  # min_mask
    ]).astype(np.float32)

    return x_cont_all, scalers


# ---------------------------------------------------------------------------
# Induced subgraph per split
# ---------------------------------------------------------------------------

def build_split_data(
    split_idx_np: np.ndarray,
    N: int,
    x_cont_all: np.ndarray,
    edge_index_full_np: np.ndarray,
    y_arrays: dict,
    device: torch.device,
) -> Data:
    """
    Build a torch_geometric.Data for a single split (train/val/test).

    Parameters
    ----------
    split_idx_np   : sorted int64 array of row indices in this split.
    N              : total number of nodes in the full graph.
    x_cont_all     : [N, 12] continuous feature array.
    edge_index_full_np : [2, E_line] full line-graph edge index (numpy).
    y_arrays       : dict with keys:
                       highway_in, lanes_in, oneway_in,
                       y_highway, y_lanes, y_oneway,
                       y_width, y_max, y_min
    device         : torch device.
    """
    split_idx_np = np.unique(split_idx_np).astype(np.int64)
    split_idx_np.sort()

    map_arr = np.full(N, -1, dtype=np.int64)
    map_arr[split_idx_np] = np.arange(len(split_idx_np), dtype=np.int64)

    src_old = edge_index_full_np[0]
    dst_old = edge_index_full_np[1]
    keep = (map_arr[src_old] >= 0) & (map_arr[dst_old] >= 0)
    src_new = map_arr[src_old[keep]]
    dst_new = map_arr[dst_old[keep]]

    edge_index = torch.from_numpy(np.stack([src_new, dst_new], axis=0)).long()

    data = Data(
        x_cont=torch.from_numpy(x_cont_all[split_idx_np]).float(),
        edge_index=edge_index,
    )
    data.num_nodes = len(split_idx_np)
    data.x = data.x_cont

    data.highway_in = torch.from_numpy(y_arrays["highway_in"][split_idx_np]).long()
    data.lanes_in = torch.from_numpy(y_arrays["lanes_in"][split_idx_np]).long()
    data.oneway_in = torch.from_numpy(y_arrays["oneway_in"][split_idx_np]).long()

    data.y_highway = torch.from_numpy(y_arrays["y_highway"][split_idx_np]).long()
    data.y_lanes = torch.from_numpy(y_arrays["y_lanes"][split_idx_np]).long()
    data.y_oneway = torch.from_numpy(y_arrays["y_oneway"][split_idx_np]).float()
    data.y_width = torch.from_numpy(y_arrays["y_width"][split_idx_np]).float()
    data.y_max = torch.from_numpy(y_arrays["y_max"][split_idx_np]).float()
    data.y_min = torch.from_numpy(y_arrays["y_min"][split_idx_np]).float()

    return data.to(device)


# ---------------------------------------------------------------------------
# Full dataset preparation pipeline
# ---------------------------------------------------------------------------

def prepare_dataset(
    final_gdf: gpd.GeoDataFrame,
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    grid_size_m: float = 1000.0,
    utm_epsg: int = 32748,
    device: torch.device | None = None,
) -> dict:
    """
    Full pipeline: GeoDataFrame -> train/val/test Data objects.

    Returns a dict with keys:
      data_train, data_val, data_test,
      hwy2id, id2hwy, HIGHWAY_MASK_ID,
      scalers, num_highway,
      train_idx, val_idx, test_idx
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = final_gdf.copy().reset_index(drop=True)
    df["eid"] = np.arange(len(df), dtype=np.int64)

    if "geometry" not in df.columns:
        raise ValueError("GeoDataFrame is missing 'geometry'.")

    df = gpd.GeoDataFrame(df, geometry="geometry")
    if df.crs is None:
        df = df.set_crs(epsg=4326)
    df = df[~df["geometry"].isna()].copy()

    df_proj = df.to_crs(epsg=utm_epsg)
    centroids = df_proj.geometry.centroid
    df["cx"] = centroids.x.astype(np.float64)
    df["cy"] = centroids.y.astype(np.float64)

    if "highway" not in df.columns:
        raise ValueError("Expected a 'highway' column.")

    # Derived columns
    df["width_m"] = df["width"]
    df["lanes_cls"] = df["lanes"].apply(lanes_to_class).astype(np.int64)
    df["max_speed_val"] = df["max_speed"]
    df["min_speed_val"] = df["min_speed"]
    df["oneway01"] = df["oneway"].astype("Float64")

    # Vocab
    hwy2id, id2hwy, HIGHWAY_MASK_ID = build_highway_vocab(df["highway"])
    highway_vals = df["highway"].fillna(UNK_TOKEN).astype(str).values
    df["highway_id"] = pd.Series(highway_vals).map(lambda x: hwy2id.get(x, hwy2id[UNK_TOKEN])).astype(np.int64)

    # Spatial split
    train_idx, val_idx, test_idx = spatial_split(
        df, seed=seed,
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        grid_size_m=grid_size_m, utm_epsg=utm_epsg,
    )

    # Normalise continuous features
    x_cont_all, scalers = normalize_features(df, train_idx)

    # Line graph
    edge_index_full = build_line_graph_edge_index(df)
    edge_index_full_np = edge_index_full.cpu().numpy()

    # Global targets / inputs
    N = len(df)
    y_highway_all = df["highway_id"].to_numpy(dtype=np.int64)
    y_lanes_all = df["lanes_cls"].to_numpy(dtype=np.int64)
    y_oneway_all = df["oneway01"].to_numpy(dtype=np.float32)
    y_width_all = df["width_m"].to_numpy(dtype=np.float32)
    y_max_all = df["max_speed_val"].to_numpy(dtype=np.float32)
    y_min_all = df["min_speed_val"].to_numpy(dtype=np.float32)

    lanes_in_all = np.where(y_lanes_all == -1, LANES_MISS_ID, y_lanes_all).astype(np.int64)
    oneway_in_all = np.where(np.isnan(y_oneway_all), ONEWAY_MISS_ID, y_oneway_all).astype(np.int64)

    y_arrays = {
        "highway_in": y_highway_all,
        "lanes_in": lanes_in_all,
        "oneway_in": oneway_in_all,
        "y_highway": y_highway_all,
        "y_lanes": y_lanes_all,
        "y_oneway": y_oneway_all,
        "y_width": y_width_all,
        "y_max": y_max_all,
        "y_min": y_min_all,
    }

    data_train = build_split_data(train_idx, N, x_cont_all, edge_index_full_np, y_arrays, device)
    data_val = build_split_data(val_idx, N, x_cont_all, edge_index_full_np, y_arrays, device)
    data_test = build_split_data(test_idx, N, x_cont_all, edge_index_full_np, y_arrays, device)

    print(f"Train: {data_train.num_nodes} nodes, {data_train.edge_index.shape[1]} edges")
    print(f"Val:   {data_val.num_nodes} nodes, {data_val.edge_index.shape[1]} edges")
    print(f"Test:  {data_test.num_nodes} nodes, {data_test.edge_index.shape[1]} edges")

    return {
        "data_train": data_train,
        "data_val": data_val,
        "data_test": data_test,
        "hwy2id": hwy2id,
        "id2hwy": id2hwy,
        "HIGHWAY_MASK_ID": HIGHWAY_MASK_ID,
        "scalers": scalers,
        "num_highway": len(hwy2id),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
