from typing import List, Optional
from app.repositories.item import ItemRepository
from app.models.item import ItemModel
from app.schemas.item import ItemCreate

class ItemService:
    """
    Handles the core business logic.
    """
    def __init__(self, repository: ItemRepository):
        self.repository = repository

    def create_item(self, data: ItemCreate) -> ItemModel:
        item = ItemModel(
            id=0,
            name=data.name,
            description=data.description,
            price=data.price
        )
        return self.repository.create(item)

    def get_item(self, item_id: int) -> Optional[ItemModel]:
        return self.repository.get_by_id(item_id)

    def get_items(self) -> List[ItemModel]:
        return self.repository.get_all()
