"""Dynamic weighted directed multigraph.

Design choices (to justify in the report):
  * Edges live in one flat list indexed by edge id -> O(1) update by id.
  * adj / radj hold edge ids per node -> forward and backward search
    (needed for bidirectional A* and for incremental re-planning).
  * Updates never delete edges; they flip `available` or change weights.
    Ids stay stable, so route caches can store edge ids safely.
  * `version` + `change_log` record every mutation, so a re-planner can ask
    "which edges changed since version v?" without rescanning the graph.
"""
from dataclasses import dataclass
from typing import Iterator, List, Tuple


@dataclass(slots=True)
class Edge:
    id: int
    u: int
    v: int
    distance: float
    time: float              # expected travel time
    time_min: float          # uncertainty bounds on travel time
    time_max: float
    fuel: float
    risk: float              # 0..1
    capacity: float
    reliability: float       # 1 - failure probability
    available: bool = True
    is_wormhole: bool = False


_MUTABLE = {"distance", "time", "time_min", "time_max", "fuel", "risk", "capacity", "reliability"}


class Graph:
    def __init__(self, n_nodes: int = 0):
        self.n = n_nodes
        self.edges: List[Edge] = []
        self.adj: List[List[int]] = [[] for _ in range(n_nodes)]
        self.radj: List[List[int]] = [[] for _ in range(n_nodes)]
        self.version = 0
        self.change_log: List[Tuple[int, int]] = []   # (version, edge_id)

    # ---- construction -------------------------------------------------
    def add_node(self) -> int:
        self.adj.append([])
        self.radj.append([])
        self.n += 1
        self.version += 1
        return self.n - 1

    def add_edge(self, u: int, v: int, distance: float, time: float, fuel: float,
                 risk: float, capacity: float, reliability: float,
                 time_min: float = None, time_max: float = None,
                 is_wormhole: bool = False) -> int:
        eid = len(self.edges)
        self.edges.append(Edge(eid, u, v, distance, time,
                               time * 0.8 if time_min is None else time_min,
                               time * 1.5 if time_max is None else time_max,
                               fuel, risk, capacity, reliability, True, is_wormhole))
        self.adj[u].append(eid)
        self.radj[v].append(eid)
        return eid

    # ---- dynamic updates (all O(1)) ----------------------------------
    def _log(self, eid: int) -> None:
        self.version += 1
        self.change_log.append((self.version, eid))

    def set_available(self, eid: int, available: bool) -> None:
        e = self.edges[eid]
        if e.available != available:
            e.available = available
            self._log(eid)

    def update_edge(self, eid: int, **fields) -> None:
        e = self.edges[eid]
        for k, val in fields.items():
            if k not in _MUTABLE:
                raise AttributeError(f"edge field '{k}' is not updatable")
            setattr(e, k, val)
        self._log(eid)

    def changed_since(self, version: int) -> List[int]:
        """Edge ids modified after `version` (deduplicated, in change order)."""
        seen, out = set(), []
        for ver, eid in reversed(self.change_log):
            if ver <= version:
                break
            if eid not in seen:
                seen.add(eid)
                out.append(eid)
        out.reverse()
        return out

    # ---- queries -------------------------------------------------------
    def out_edges(self, u: int) -> Iterator[Edge]:
        edges = self.edges
        for eid in self.adj[u]:
            e = edges[eid]
            if e.available:
                yield e

    def in_edges(self, v: int) -> Iterator[Edge]:
        edges = self.edges
        for eid in self.radj[v]:
            e = edges[eid]
            if e.available:
                yield e

    @property
    def num_edges(self) -> int:
        return len(self.edges)