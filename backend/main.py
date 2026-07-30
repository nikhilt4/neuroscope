from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()

from backend.routes.upload import router as upload_router


#FastAPI app instance
app = FastAPI(title = "AI1 API")

#CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5500", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.include_router(upload_router)

@app.get("/")
async def root():
    return {"status": "NeuroScope API running"}