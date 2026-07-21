from fastapi import APIRouter, Depends
from app.controllers.item import ItemController
from app.services.item import ItemService
from app.repositories.item import ItemRepository
from app.schemas.item import ItemCreate, ItemResponse
from typing import List

router = APIRouter(prefix="/items", tags=["items"])

def get_item_repository() -> ItemRepository:
    if not hasattr(get_item_repository, "repo"):
        get_item_repository.repo = ItemRepository()
    return get_item_repository.repo

def get_item_service(repo: ItemRepository = Depends(get_item_repository)) -> ItemService:
    return ItemService(repo)

def get_item_controller(service: ItemService = Depends(get_item_service)) -> ItemController:
    return ItemController(service)

@router.post("/", response_model=ItemResponse)
def create_item(item: ItemCreate, controller: ItemController = Depends(get_item_controller)):
    return controller.create_item(item)

@router.get("/{item_id}", response_model=ItemResponse)
def get_item(item_id: int, controller: ItemController = Depends(get_item_controller)):
    return controller.get_item(item_id)

@router.get("/", response_model=List[ItemResponse])
def get_items(controller: ItemController = Depends(get_item_controller)):
    return controller.get_items()
