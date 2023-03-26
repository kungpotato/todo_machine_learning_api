from fastapi import FastAPI
from app.api.todo_api import app as todo_router

app = FastAPI()

app.include_router(todo_router, prefix="/todo", tags=["todo"])
