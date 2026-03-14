from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import jwt

from .schemas import InferenceRequest, InferenceResponse
from .services.features import extract_features
from .services.scoring import score_features

app = FastAPI(title="PassiveCaptcha Enterprise API", version="1.0.0")

# ── Phase 5: Enterprise Hardening Variables ──
SECRET_KEY = "super-secret-enterprise-key"
ALGORITHM = "HS256"

# Mock Redis-like session store + Rate limiting
redis_store = {}
rate_limits = {}
RATE_LIMIT_MAX_REQUESTS = 120 # Increased to allow constant dashboard polling
RATE_LIMIT_WINDOW_SEC = 60

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["session_id"]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token validation failed")

def enforce_rate_limit(request: Request):
    client_ip = request.client.host
    now = datetime.now()
    
    if client_ip not in rate_limits:
        rate_limits[client_ip] = []
        
    # Clean up old requests
    rate_limits[client_ip] = [ts for ts in rate_limits[client_ip] if now - ts < timedelta(seconds=RATE_LIMIT_WINDOW_SEC)]
    
    if len(rate_limits[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Too many requests.")
        
    rate_limits[client_ip].append(now)

@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/v1/init")
def init_session(request: Request):
    enforce_rate_limit(request)
    session_id = f"sess_{int(datetime.now().timestamp())}"
    token = jwt.encode({"session_id": session_id, "exp": datetime.utcnow() + timedelta(hours=1)}, SECRET_KEY, algorithm=ALGORITHM)
    redis_store[session_id] = {"init_time": datetime.utcnow().isoformat(), "requests": 0}
    return {"token": token, "session_id": session_id}

@app.post("/v1/ingest", response_model=InferenceResponse)
def ingest_session(payload: InferenceRequest, request: Request, session_id: str = Depends(verify_token)) -> InferenceResponse:
    enforce_rate_limit(request)
    
    # Update Redis mock store
    if session_id in redis_store:
        redis_store[session_id]["requests"] += 1
    
    features = extract_features(payload)
    # Pass the raw payload so score_features can forward mouse sequences to the LSTM
    return score_features(features, payload)
