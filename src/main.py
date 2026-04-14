from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from chat import get_response, bot_name

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Message(BaseModel):
    text: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat")
def chat(message: Message):
    response = get_response(message.text)
    return {"sender": bot_name, "response": response}
