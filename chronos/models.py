"""Domain models. Plain dataclasses with __slots__ to keep memory low at 100k+ planets.

Per-resource quantities are lists indexed by Resource (O(1) access, compact).
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Set


class Resource(IntEnum):
    FOOD = 0
    WATER = 1
    ENERGY = 2
    FUEL = 3
    MEDICINE = 4
    RARE_MINERALS = 5
    CONSTRUCTION = 6


NUM_RESOURCES = len(Resource)


class NodeKind(IntEnum):
    PLANET = 0
    COLONY = 1
    STATION = 2


@dataclass(slots=True)
class Planet:
    id: int
    x: float
    y: float
    kind: NodeKind
    population: int
    technology: int          # 1..10
    defense: int             # 0..10
    importance: float        # strategic importance, 0..1
    production: List[float]  # units/day per resource
    consumption: List[float]  # units/day per resource
    storage_cap: List[float]
    stock: List[float]


@dataclass(slots=True)
class Spacecraft:
    id: int
    location: int
    cargo_capacity: float
    fuel_capacity: float
    fuel: float
    speed: float
    available_time: float
    reliability: float
    operating_cost: float            # cost per unit distance
    capabilities: Set[str] = field(default_factory=set)
    alive: bool = True


@dataclass(slots=True)
class Mission:
    id: int
    origin: int
    destination: int
    cargo_type: Resource
    amount: float
    release_time: float
    deadline: float
    priority: int                    # 1 (low) .. 5 (emergency)
    importance: float
    required_capability: str = "cargo"


class EventType(IntEnum):
    WORMHOLE_COLLAPSE = 0
    ROUTE_BLOCKED = 1
    PIRATE_BLOCKADE = 2
    SHIP_DESTROYED = 3
    EMERGENCY_MEDICAL = 4
    PRODUCTION_DROP = 5
    NEW_COLONY = 6
    POPULATION_SURGE = 7
    RESOURCE_UNAVAILABLE = 8
    HIGH_PRIORITY_MISSION = 9
    ROUTE_DANGER_UP = 10
    SHIP_MECHANICAL_FAILURE = 11


@dataclass(slots=True)
class Event:
    time: float
    type: EventType
    target: int          # edge id / planet id / ship id depending on type (-1 = none)
    severity: float      # 0..1


@dataclass(slots=True)
class FacilityOption:
    """A candidate infrastructure project (one knapsack/facility-location item)."""
    id: int
    planet: int
    kind: str
    cost: float
    build_time: float
    resource: Optional[Resource]
    resource_delta: float     # units/day produced (+) or consumed (-)
    population_delta: int
    energy_delta: float
    strategic_value: float