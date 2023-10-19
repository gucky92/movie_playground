from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from data_tools import DataTools


app = FastAPI()
tools = DataTools(overwrite=False)


class MovieRec(BaseModel):
    query: str


@app.api_route('/', methods=['GET', 'POST'])
async def root(movie: MovieRec = None):
    if movie is None:
        return {
            "assistant": "Please provide a query to recommend a movie.", 
            "recommendations": None,
        }
    
    list_of_titles = tools.retrieve(movie.query)
    
    if isinstance(list_of_titles, str):
        return {
            "assistant": list_of_titles, 
            "recommendations": None,
        }
    return {
        "recommendations": list_of_titles, 
        "assistant": None,
    }