from fastapi import FastAPI
from app.routers.item import router as item_router
from app.routers.recommendation import router as recommendation_router
from app.routers.technical import router as technical_router

app = FastAPI(title="Structured FastAPI Architecture")

app.include_router(item_router)
app.include_router(recommendation_router)
app.include_router(technical_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the structured FastAPI App. Go to /docs for the API swagger."}
