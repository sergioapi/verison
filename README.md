# VeriSon

VeriSon is a full-stack web application that analyzes a music track and estimates whether it is more likely to be **AI-generated** or **human-created**.

**[Try the live demo](https://verison-app.vercel.app)**

> The public demo uses Vercel for the frontend and Render for the backend. The free backend instance may need a short cold start after inactivity.

## How it works

The React/TypeScript frontend sends the uploaded audio to a FastAPI backend, where the track is processed using an **MFCC + SVM** inference pipeline before returning the estimated origin.

## Tech stack

**Frontend:** React, TypeScript, Vite  
**Backend:** Python, FastAPI, scikit-learn, librosa  
**Testing:** pytest, Vitest, Testing Library  
**Tooling:** Docker, GitHub Actions

## Run locally

### Backend

From the repository root:

```bash
docker build -f backend/Dockerfile -t verison-backend .
docker run --rm -p 8000:8000 -e CORS_ALLOWED_ORIGINS=http://localhost:5173 verison-backend
```

API documentation:

```text
http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm ci
```

Create `frontend/.env` with:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Then run:

```bash
npm run dev
```

Open `http://localhost:5173`.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/v1/analyze` | Analyze an uploaded audio file |
| `GET` | `/api/v1/model` | Get model metadata |
| `GET` | `/health` | Service health |
| `GET` | `/ready` | Inference readiness |

## Tests and CI

```bash
# Backend
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests

# Frontend
cd frontend
npm ci
npm run check
```

GitHub Actions runs backend tests, frontend checks and a Docker build on pushes and pull requests to `main`.
