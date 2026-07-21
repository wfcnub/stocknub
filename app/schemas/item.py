from pydantic import BaseModel
from typing import Optional

class ItemCreate(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
