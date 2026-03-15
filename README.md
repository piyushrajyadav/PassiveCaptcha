# PassiveCaptcha: ML-Based Passive Human Verification System

PassiveCaptcha is an intelligent, machine learning-driven system designed to act as a seamless alternative to traditional CAPTCHAs. It captures passive behavioral and environmental parameters�such as mouse dynamics, typing rhythms, scroll patterns, and hardware entropy�to differentiate between human users and automated bots in real time, eliminating user friction while securing enterprise applications.

## Key Features
- **Passive Environmental Analysis**: Detects hardware characteristics, WebGL renderer availability, pixel ratios, and viewport anomalies commonly seen in headless automated sessions.
- **Behavioral Intelligence ML**: Ingests high-frequency signals (mouse trajectories, click intervals, scroll rates) to a Python FastApi backend.
- **Real-Time Classification**: Utilizes an ensemble ML backend (XGBoost, Random Forests, LSTM) to output a confidence interval: `human`, `bot`, or `review`.
- **Low-Friction Fallback (Phase 3)**: If the backend expresses low confidence (`review`), the SDK triggers a lightweight, minimal interactive challenge (e.g., clicking a moving dot) rather than a complex image-labeling puzzle.
- **Enterprise Analytics Dashboard (Phase 4)**: A React-based diagnostic UI displaying live streaming diagnostics, feature importance (SHAP values), flagged signal timelines, and historical confidence scores.
- **API Hardening & SDK (Phase 5)**: Enterprise-grade security out-of-the-box:
  - **Rate Limiting**: Protects against rapid verification abuse.
  - **JWT Session Tokens**: Secures telemetry streams tying ingest calls strictly to an initiated, monitored session.
  - **Session Persistence**: Extensible caching strategy (Redis-ready mock included) resolving temporal bot profiles over time.
  - **`PassiveCaptcha.init()` SDK**: Clean, pluggable JS snippet allowing instantaneous client integration.

## Folder Structure
```text
PassiveCaptcha/
+-- apps/
�   +-- web/                 # React UI, Live Analytics Dashboard, & Fallback Challenge
�       +-- sdk.js           # Client-side enterprise wrapper
�       +-- src/
�       �   +-- App.tsx      # Main application dashboard
�       �   +-- Challenge.tsx# Minimal fallback UI component
�       �   +-- Dashboard.tsx# Live Analytics monitoring UI
�       �   +-- lib/         # Signal collection logic & ML inferences logic
�       +-- package.json
�       +-- vite.config.ts
+-- backend/                 # FastAPI & ML Inference Engine
�   +-- app/
�   �   +-- main.py          # FastApi server (JWT Auth, Rate limiting configured)
�   �   +-- services/        # ML Extractors & Model Serving 
�   �   +-- schemas.py
�   +-- train/               # Jupyter/Python scripts for ML training (LSTM + XGBoost)
�   +-- requirements-api.txt # API layer Dependencies
�   +-- requirements-ml.txt  # Training & Inference Dependencies
+-- docker-compose.yml       # Production-ready orchestration
+-- package.json             # Monorepo command runner
```

## System Architecture

1. **Client-Side (SDK & Signal Collector)**
   The client runs a silent, passive collector. Information is batched into JSON snapshots detailing pointer geometry, keystroke cadence, and deep hardware checks. 

2. **API & Data Ingest Layer (FastAPI)**
   The backend provides `POST /v1/init` and `POST /v1/ingest`. `init` bootstraps a JWT session and prepares a temporal Redis window. As telemetry flows to `ingest`, it enforces rate limits, validates tokens, and forwards to the inference pipeline.

3. **Inference pipeline**
   We extract standard deviations, acceleration max rates, and path variance. The feature vector passes through an ensemble:
   - *Gradient Boosted Decision Trees (XG)*: For broad structural classification.
   - *LSTM*: For sequence modeling (examining mouse X-Y coordinates over time).

4. **Response & Orchestration**
   The response contains `verdict` & `score`. If the client detects `verdict === "review"`, the `ChallengeOverlay` triggers. Otherwise, the interaction propagates uninterrupted.

## How to Run locally

### Prerequisites
- Node.js (v16+)
- Python (3.9+)

### Installation

1. **Install Frontend Dependencies:**
   ```bash
   cd apps/web
   npm install
   ```

2. **Install Backend Dependencies:**
   ```bash
   # Use a virtual environment
   cd backend
   pip install -r requirements-api.txt
   ```

### Start Development Servers

1. **Start the API Backend:**
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the Frontend Dashboard:**
   ```bash
   cd apps/web
   npm run dev
   ```
   Navigate to `http://localhost:5173` to see the live Analytics dashboard in action, observing your own session. Emulate simple automated-like behavior (e.g., refreshing repeatedly or moving mouse uniformly) to trigger the `review` fallback challenge.

## Production Deployment (Docker)
The repository contains a `docker-compose.yml` for unified scaling deployment.

```bash
docker-compose up --build -d
```

## Privacy & Compliance
PassiveCaptcha operates strictly without tracking persistent cookies or capturing deeply sensitive input field keystrokes. Interactions are heavily anonymized into numeric deltas and geometric deviations before leaving the browser, ensuring full GDPR and CCPA compliance.

## Open Innovation & Future Extensibility
To extend beyond standard CAPTCHA replacement paradigms, this solution incorporates the following innovative concepts:
1. **Explainable AI (XAI)**: We compute SHAP (SHapley Additive exPlanations) values on the fly to provide a transparent "Feature Breakdown" panel. Enterprises know *why* a session was flagged (e.g., "Abnormal Scroll Velocity").
2. **Federated Threat Intelligence Scope**: Built on a modular architecture, the backend allows for easy integration with global IP threat feeds or enterprise SIEM platforms.
3. **Continuous Reinforcement Loop**: Real-world challenge results (User solved dot-click vs Abandoned) can be automatically ingested back into the data pipelines to re-train the XGBoost models iteratively.

## Authors
Created by the ML Security Automation Team as an alternative defense schema for digital enterprise onboarding and validation.
