"""
Load a road network graph from the Mapedia database (DBHandler).
"""
import sys
from pathlib import Path


def load_graph_from_db(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    repo_root: str | Path | None = None,
) -> object:
    """
    Connect to the Mapedia database and fetch the road network graph
    for the given bounding box.

    Parameters
    ----------
    min_lat, max_lat : float  Latitude bounds.
    min_lon, max_lon : float  Longitude bounds.
    repo_root : str | Path, optional
        Path to the mapedia repo root so that `modules` is importable.
        Defaults to three levels above this file.

    Returns
    -------
    networkx.MultiDiGraph
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo_root))
    sys.modules.pop("modules", None)

    from modules import DBHandler  # noqa: PLC0415

    db_handler = DBHandler()
    db_handler.connect_to_db()

    G = db_handler.get_graph(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )
    print(f"Loaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G
