import dataclasses
import json


@dataclasses.dataclass
class Point:
    x : float = 0.0
    y: float = 0.0


@dataclasses.dataclass
class Params:
    x : int = 0
    y: float = 0.0
    z: bool = True
    L: list = dataclasses.field(default_factory=lambda: [1, 2, 3])
    P: Point = Point()


p = Params()
s = json.dumps(dataclasses.asdict(p))
print(s)
p2 = Params(**json.loads(s))
print(p2)