from fastapi import FastAPI
from pydantic import BaseModel

from src.main import ModelMain

app = FastAPI()

class textchunk(BaseModel):
    paragraph: str

@app.post('/')
async def scoring_endpoint(para: textchunk):
    Model = ModelMain(para.paragraph)
    return Model.runnerClass()
