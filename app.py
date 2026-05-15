"""FastAPI entrypoint — run: python app.py  (opens at http://localhost:8501)"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="127.0.0.1", port=8501, reload=True)
