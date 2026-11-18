"""Reproducible galaxy generator for Small / Medium / Large / Extreme datasets.

Generation is O(n + m):
  1. Nodes get random 2D coordinates and are bucketed in a uniform grid.
  2. A "snake" walk over the grid links consecutive nodes -> guaranteed
     connected backbone with mostly short, local edges.
  3. Extra routes join random nodes in neighbouring cells (local mesh).
  4. A few long-range wormholes (very fast, costly in fuel, riskier).
Every undirected route becomes two directed arcs, so
  num_routes (directed arcs) = 2 * number of generated pairs.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .graph import Graph
from .models import (EventType, Event, FacilityOption, Mission, NodeKind,
                     NUM_RESOURCES, Planet, Resource, Spacecraft)

SIZES: Dict[str, Tuple[int, int, int]] = {      # planets, routes (arcs), missions
    "small": (100, 500, 50),
    "medium": (1_000, 10_000, 500),
    "large": (10_000, 100_000, 5_000),
    "extreme": (100_000, 1_000_000, 50_000),
}
SIDE = 1000.0   # galaxy is a SIDE x SIDE square

# per-capita daily consumption for each Resource (index = Resource value)
PER_CAPITA = [0.02, 0.03, 0.015, 0.008, 0.002, 0.001, 0.004]

FACILITY_CATALOG = {
    # kind: (cost, build_time, resource, resource_delta, pop_delta, energy_delta, strategic)
    "energy_plant": (500, 20, Resource.ENERGY, 120.0, 0, 0.0, 3.0),
    "farm": (200, 10, Resource.FOOD, 80.0, 0, -10.0, 2.0),
    "hospital": (350, 15, Resource.MEDICINE, 30.0, 500, -15.0, 4.0),
    "factory": (600, 25, Resource.CONSTRUCTION, 50.0, 0, -40.0, 3.0),
    "defense_system": (450, 18, None, 0.0, 0, -25.0, 6.0),
    "research_station": (400, 30, None, 0.0, 0, -20.0, 5.0),
    "spaceport": (700, 22, Resource.FUEL, 40.0, 0, -30.0, 5.0),
    "mining_facility": (550, 20, Resource.RARE_MINERALS, 25.0, 0, -35.0, 4.0),
}


@dataclass
class Galaxy:
    graph: Graph
    planets: List[Planet]
    ships: List[Spacecraft]
    missions: List[Mission]
    facilities: List[FacilityOption]
    events: List[Event]
    budget: float
    seed: int
    name: str = "custom"


def _make_planets(rng: random.Random, n: int) -> Tuple[List[Planet], List[Tuple[float, float]]]:
    planets, coords = [], []
    for i in range(n):
        x, y = rng.uniform(0, SIDE), rng.uniform(0, SIDE)
        coords.append((x, y))
        r = rng.random()
        kind = NodeKind.PLANET if r < 0.6 else NodeKind.COLONY if r < 0.9 else NodeKind.STATION
        pop = int(rng.lognormvariate(8.5, 1.0))
        if kind == NodeKind.STATION:
            pop //= 10
        prod, cons, cap, stock = [], [], [], []
        for k in range(NUM_RESOURCES):
            c = pop * PER_CAPITA[k] * rng.uniform(0.9, 1.1)
            # production covers 60%..130% of consumption -> some planets run deficits
            p = c * rng.uniform(0.6, 1.3)
            capacity = c * rng.uniform(30, 90)
            prod.append(p); cons.append(c); cap.append(capacity)
            stock.append(capacity * rng.uniform(0.2, 0.9))
        planets.append(Planet(i, x, y, kind, pop, rng.randint(1, 10), rng.randint(0, 10),
                              rng.random(), prod, cons, cap, stock))
    return planets, coords


def _build_graph(rng: random.Random, coords, n_pairs: int) -> Graph:
    n = len(coords)
    g = Graph(n)
    cps = max(1, int(math.sqrt(n / 2)))              # cells per side (~2 nodes/cell)
    cell = SIDE / cps
    grid: List[List[List[int]]] = [[[] for _ in range(cps)] for _ in range(cps)]
    for i, (x, y) in enumerate(coords):
        grid[min(cps - 1, int(y / cell))][min(cps - 1, int(x / cell))].append(i)

    pairs = set()

    def add_pair(u: int, v: int, wormhole: bool = False) -> bool:
        if u == v:
            return False
        key = (u, v) if u < v else (v, u)
        if key in pairs:
            return False
        pairs.add(key)
        d = math.dist(coords[u], coords[v])
        if wormhole:
            t, fuel, risk, rel = d * 0.05, d * 0.3, rng.uniform(0.10, 0.30), rng.uniform(0.85, 0.95)
        else:
            t = d * rng.uniform(0.8, 1.2)
            fuel = d * rng.uniform(1.0, 2.0)
            risk = min(0.5, rng.betavariate(1.2, 12))
            rel = max(0.5, 1 - risk * rng.uniform(0.1, 0.5))
        cap = rng.choice([100, 250, 500, 1000])
        for a, b in ((u, v), (v, u)):
            g.add_edge(a, b, d, t, fuel, risk, cap, rel,
                       time_min=t * 0.8, time_max=t * rng.uniform(1.3, 1.8), is_wormhole=wormhole)
        return True

    # 1) backbone: snake walk over the grid -> connected
    order: List[int] = []
    for cy in range(cps):
        xs = range(cps) if cy % 2 == 0 else range(cps - 1, -1, -1)
        for cx in xs:
            order.extend(grid[cy][cx])
    for a, b in zip(order, order[1:]):
        add_pair(a, b)

    # 2) wormholes (long range)
    n_worm = min(max(1, n_pairs // 100), max(0, n_pairs - len(pairs)))
    attempts = 0
    while n_worm > 0 and attempts < 50 * n_worm + 50:
        attempts += 1
        if add_pair(rng.randrange(n), rng.randrange(n), wormhole=True):
            n_worm -= 1

    # 3) local mesh until we reach the requested number of pairs
    attempts, limit = 0, 30 * max(0, n_pairs - len(pairs)) + 100
    while len(pairs) < n_pairs and attempts < limit:
        attempts += 1
        u = rng.randrange(n)
        cx = min(cps - 1, int(coords[u][0] / cell)); cy = min(cps - 1, int(coords[u][1] / cell))
        nx = cx + rng.randint(-1, 1); ny = cy + rng.randint(-1, 1)
        if 0 <= nx < cps and 0 <= ny < cps and grid[ny][nx]:
            add_pair(u, rng.choice(grid[ny][nx]))
    return g


def _make_ships(rng, n_ships: int, n_planets: int) -> List[Spacecraft]:
    ships = []
    for i in range(n_ships):
        fuel_cap = rng.uniform(800, 4000)
        caps = {"cargo"}
        if rng.random() < 0.3: caps.add("medical")
        if rng.random() < 0.2: caps.add("armed")
        ships.append(Spacecraft(i, rng.randrange(n_planets), rng.uniform(100, 1000), fuel_cap,
                                fuel_cap * rng.uniform(0.5, 1.0), rng.uniform(0.8, 2.0), 0.0,
                                rng.uniform(0.9, 0.995), rng.uniform(0.5, 3.0), caps))
    return ships


def _make_missions(rng, n_missions: int, n_planets: int, horizon: float) -> List[Mission]:
    ms = []
    for i in range(n_missions):
        o, d = rng.randrange(n_planets), rng.randrange(n_planets)
        while d == o:
            d = rng.randrange(n_planets)
        rtype = Resource(rng.randrange(NUM_RESOURCES))
        prio = rng.choices([1, 2, 3, 4, 5], weights=[30, 30, 20, 15, 5])[0]
        release = rng.uniform(0, horizon * 0.8)
        cap = "medical" if rtype == Resource.MEDICINE and rng.random() < 0.7 else "cargo"
        ms.append(Mission(i, o, d, rtype, rng.uniform(20, 800), release,
                          release + rng.uniform(100, 600) / (1 + 0.3 * (prio - 1)),
                          prio, rng.random(), cap))
    return ms


def _make_facilities(rng, n_planets: int, count: int) -> Tuple[List[FacilityOption], float]:
    opts = []
    kinds = list(FACILITY_CATALOG)
    for i in range(count):
        kind = rng.choice(kinds)
        cost, bt, res, dres, dpop, den, sv = FACILITY_CATALOG[kind]
        opts.append(FacilityOption(i, rng.randrange(n_planets), kind,
                                   cost * rng.uniform(0.8, 1.3), bt, res, dres,
                                   dpop, den, sv * rng.uniform(0.7, 1.3)))
    return opts, 0.3 * sum(o.cost for o in opts)       # budget funds ~30% of options


def _make_events(rng, n_events: int, horizon: float, g: Graph, n_planets: int, n_ships: int) -> List[Event]:
    evs = []
    wormholes = [e.id for e in g.edges if e.is_wormhole] or [0]
    for _ in range(n_events):
        t = rng.uniform(0.05 * horizon, horizon)
        et = EventType(rng.randrange(len(EventType)))
        if et == EventType.WORMHOLE_COLLAPSE:
            target = rng.choice(wormholes)
        elif et in (EventType.ROUTE_BLOCKED, EventType.PIRATE_BLOCKADE, EventType.ROUTE_DANGER_UP):
            target = rng.randrange(g.num_edges)
        elif et in (EventType.SHIP_DESTROYED, EventType.SHIP_MECHANICAL_FAILURE):
            target = rng.randrange(max(1, n_ships))
        elif et in (EventType.NEW_COLONY, EventType.HIGH_PRIORITY_MISSION):
            target = -1
        else:
            target = rng.randrange(n_planets)
        evs.append(Event(t, et, target, rng.uniform(0.1, 1.0)))
    evs.sort(key=lambda e: e.time)
    return evs


def generate(size="small", seed: int = 1, horizon: float = 365.0) -> Galaxy:
    """size: 'small'|'medium'|'large'|'extreme' or a (planets, routes, missions) tuple."""
    name = size if isinstance(size, str) else "custom"
    n, m, k = SIZES[size] if isinstance(size, str) else size
    rng = random.Random(seed)
    planets, coords = _make_planets(rng, n)
    graph = _build_graph(rng, coords, m // 2)
    ships = _make_ships(rng, max(5, k // 5), n)
    missions = _make_missions(rng, k, n, horizon)
    facilities, budget = _make_facilities(rng, n, max(10, n // 10))
    events = _make_events(rng, max(5, k // 5), horizon, graph, n, len(ships))
    return Galaxy(graph, planets, ships, missions, facilities, events, budget, seed, name)