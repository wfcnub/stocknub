from dataclasses import dataclass
from typing import Optional

@dataclass
class ItemModel:
    id: int
    name: str
    description: Optional[str] = None
    price: float = 0.0
