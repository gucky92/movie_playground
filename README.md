# movie_playground
[WIP] Testing LLMs for semantic search of movie databases

This project implements a "semantic search" over a movie database using LLMs like GPT-3.

Features
Loads movie metadata like title, plot, cast etc from a parquet file
Enriches metadata by scraping Wikipedia infoboxes
Stores movies as nodes in a Chroma vector database
Implements a DataTools class that can retrieve movies using natural language queries
Exposes the search via a simple FastAPI web app
