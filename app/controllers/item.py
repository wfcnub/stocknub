from fastapi import HTTPException
from typing import List
from app.services.item import ItemService
from app.schemas.item import ItemCreate, ItemResponse

class ItemController:
    """
    Handles HTTP-specific logic, extracting request data, calling service, and preparing response.
    """
    def __init__(self, service: ItemService):
        self.service = service

    def create_item(self, item_data: ItemCreate) -> ItemResponse:
        created_item = self.service.create_item(item_data)
        return ItemResponse(**created_item.__dict__)

    def get_item(self, item_id: int) -> ItemResponse:
        item = self.service.get_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return ItemResponse(**item.__dict__)

    def get_items(self) -> List[ItemResponse]:
        items = self.service.get_items()
        return [ItemResponse(**item.__dict__) for item in items]
