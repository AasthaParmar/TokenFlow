from fastapi import FastAPI

app = FastAPI(title="TokenFlow", description="LLM efficiency gateway")


@app.get("/health")
def health():
    return {"status": "ok", "service": "tokenflow"}
