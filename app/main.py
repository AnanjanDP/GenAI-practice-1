
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.search import SemanticSearchEngine


# 1. Define request and response models first

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)


class SearchResult(BaseModel):
    id: int
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResult]


# 2. Initialize the search engine at application startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = SemanticSearchEngine()
    yield


app = FastAPI(
    title="Semantic Search Engine",
    description="Search technical documents by meaning.",
    version="1.0.0",
    lifespan=lifespan,
)


# 3. Define API endpoints

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        results = app.state.engine.search(
            request.query,
            k=request.k,
        )

        return {
            "query": request.query,
            "count": len(results),
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc