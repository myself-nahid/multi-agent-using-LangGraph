from fastapi import FastAPI, Request

app = FastAPI(title="SARA-Multi-Agent AI Platform")

@app.get("/")
def read_root():
    return {"Hello": "World"}
