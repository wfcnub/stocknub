from typing import List, Optional
from app.models.item import ItemModel

class ItemRepository:
    """
    Simulates a database interaction layer.
    """
    def __init__(self):
        self._db = {}
        self._current_id = 1

    def create(self, item: ItemModel) -> ItemModel:
        item.id = self._current_id
        self._db[self._current_id] = item
        self._current_id += 1
        return item

    def get_by_id(self, item_id: int) -> Optional[ItemModel]:
        return self._db.get(item_id)

    def get_all(self) -> List[ItemModel]:
        return list(self._db.values())
