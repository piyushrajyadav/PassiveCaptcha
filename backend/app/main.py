from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import InferenceRequest, InferenceResponse
from .services.features import extract_features
from .services.scoring import score_features

app = FastAPI(title="PassiveCaptcha API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ingest", response_model=InferenceResponse)
def ingest_session(payload: InferenceRequest) -> InferenceResponse:
    features = extract_features(payload)
    return score_features(features)
