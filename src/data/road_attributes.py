"""
Convert a road-network graph to a GeoDataFrame and merge road attributes
fetched from the Mapedia PostgreSQL database.

Main entry point:  build_final_gdf(G, ...)
"""
import ast

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from sqlalchemy import create_engine


# ---------------------------------------------------------------------------
# Graph -> GeoDataFrame
# ---------------------------------------------------------------------------

def graph_to_edge_gdf(G) -> gpd.GeoDataFrame:
    """Convert a MultiDiGraph to an edge GeoDataFrame with u, v, key as columns."""
    _, gdf_edges = ox.graph_to_gdfs(G)
    gdf_edges = gdf_edges.reset_index()
    return gdf_edges


# ---------------------------------------------------------------------------
# Database fetch
# ---------------------------------------------------------------------------

def _make_engine(db_host: str, db_name: str, db_user: str, db_pass: str, db_port: int = 5432):
    return create_engine(f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")


def fetch_road_attributes_in_chunks(
    engine,
    osmids: list,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Fetch road_attributes rows for the given OSM IDs, in batches."""
    out = []
    for i in range(0, len(osmids), chunk_size):
        chunk = osmids[i : i + chunk_size]
        ids_tuple = str(tuple(chunk)).replace(",)", ")")
        q = f"""
            SELECT osm_id, road_type, width, nlanes, max_speed, min_speed
            FROM road_attributes
            WHERE osm_id IN {ids_tuple}
        """
        out.append(pd.read_sql(q, engine))
        print(f"Fetched road_attributes chunk {i}..{min(i + chunk_size, len(osmids))}")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _parse_item_as_float(item) -> float:
    s = str(item).lower().replace("m", "").strip()
    if s in {"nan", "none", ""}:
        return np.nan
    if ";" in s:
        s = s.split(";")[0].strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def clean_width_db(v) -> float:
    if pd.isna(v):
        return np.nan
    if isinstance(v, (list, tuple)):
        vals = [_parse_item_as_float(x) for x in v]
        vals = [x for x in vals if not np.isnan(x)]
        return float(np.mean(vals)) if vals else np.nan
    s = str(v).strip()
    if s.startswith("["):
        try:
            lst = ast.literal_eval(s)
            vals = [_parse_item_as_float(x) for x in lst]
            vals = [x for x in vals if not np.isnan(x)]
            return float(np.mean(vals)) if vals else np.nan
        except Exception:
            return np.nan
    return _parse_item_as_float(v)


def clean_nlanes_db(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s.startswith("["):
        try:
            lst = ast.literal_eval(s)
            return int(_parse_item_as_float(lst[0]))
        except Exception:
            return np.nan
    x = _parse_item_as_float(v)
    return int(x) if not np.isnan(x) else np.nan


def clean_oneway_edge(v):
    if pd.isna(v):
        return pd.NA
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    s = str(v).strip().lower()
    if s in {"yes", "true", "1"}:
        return 1
    if s in {"-1", "reverse", "reversed"}:
        return 1
    if s in {"no", "false", "0"}:
        return 0
    if s in {"unknown", "nan", "none", "", "reversible", "alternating"}:
        return pd.NA
    try:
        x = float(s)
        if x == -1.0:
            return 1
        if x == 0.0:
            return 0
        if x == 1.0:
            return 1
        return pd.NA
    except Exception:
        return pd.NA


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_final_gdf(
    G,
    db_host: str,
    db_name: str,
    db_user: str,
    db_pass: str,
    db_port: int = 5432,
) -> gpd.GeoDataFrame:
    """
    Full pipeline: graph -> GeoDataFrame -> merge road_attributes -> clean.

    Returns a GeoDataFrame with columns:
        u, v, key, id, osmid, oneway, highway, lanes, width,
        length, geometry, max_speed, min_speed
    """
    gdf_edges = graph_to_edge_gdf(G)
    print(f"Edges: {len(gdf_edges)}, CRS: {gdf_edges.crs}")

    engine = _make_engine(db_host, db_name, db_user, db_pass, db_port)
    osmids = gdf_edges["osmid"].astype(np.int64).unique().tolist()
    print(f"Unique osmids: {len(osmids)}")

    ra = fetch_road_attributes_in_chunks(engine, osmids)
    print(f"road_attributes rows: {len(ra)}, unique osm_id: {ra['osm_id'].nunique()}")

    ra = ra.rename(columns={"osm_id": "osmid"})
    merged = gdf_edges.merge(ra, on="osmid", how="left", validate="many_to_one")

    final_gdf = merged.copy()
    final_gdf["oneway"] = final_gdf["oneway"].apply(clean_oneway_edge).astype("Int64")
    final_gdf["width"] = final_gdf["width"].apply(clean_width_db)
    final_gdf["lanes"] = final_gdf["nlanes"].apply(clean_nlanes_db)
    final_gdf["length"] = pd.to_numeric(final_gdf["length"], errors="coerce")
    final_gdf["max_speed"] = pd.to_numeric(final_gdf["max_speed"], errors="coerce")
    final_gdf["min_speed"] = pd.to_numeric(final_gdf["min_speed"], errors="coerce")

    keep_cols = [
        "u", "v", "key", "id", "osmid", "oneway", "road_type",
        "lanes", "width", "length", "geometry", "max_speed", "min_speed",
    ]
    missing = [c for c in keep_cols if c not in final_gdf.columns]
    if missing:
        raise ValueError(f"Missing expected columns after merge/clean: {missing}")

    final_gdf = gpd.GeoDataFrame(final_gdf[keep_cols], geometry="geometry", crs=gdf_edges.crs)
    final_gdf = final_gdf.rename(columns={"road_type": "highway"})

    print("\nMissing values:")
    print(final_gdf[["highway", "lanes", "width", "oneway", "length", "max_speed", "min_speed"]].isnull().sum())
    return final_gdf
