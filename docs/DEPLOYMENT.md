# Deployment and operations

Live: https://evalforge-024b.onrender.com

## Public walkthrough

Deploy one Python web service from the main branch:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
- Environment: `EVALFORGE_MODE=walkthrough`, `PYTHON_VERSION=3.11.11`
- HTTP health path: `/health` (200 only when the configured mode is ready)

Alternatively use the included non-root Docker image. No database or secret is required. The application serves its UI and API from one origin. Free Render instances can sleep; first requests may take longer. A free-tier demo has no uptime guarantee.

After a deploy is live, verify `/`, `/health`, `/examples`, `/results`, and a POST to `/generate-sql` using a listed example. Test unsupported questions return 422, rather than fabricated output. Inspect build and runtime logs; preserve the deployed Git commit. Roll back by redeploying the last known working commit. There is no persistent application data to migrate.

## Adapter inference

Use a machine with adequate model memory, install `requirements-ml.txt`, and set `EVALFORGE_MODE=model` plus `ADAPTER_MODEL` to a real adapter directory or Hub ID. Export `USE_4BIT=true` on CUDA for quantization. This is not suitable for the lightweight free walkthrough instance. Private adapters require a securely configured `HF_TOKEN`.

Readiness is established only after loading the model. One inference runs at a time; concurrent requests receive 429 with Retry-After. Input lengths are bounded. Failures return a generic error and log the exception on the server. The demo does not store submitted schemas or execute generated SQL.

Before opening unrestricted model inference publicly, put authentication, per-user rate limits, infrastructure timeouts, and resource monitoring at the gateway. The current service is intended for a small portfolio audience, not an SLA-backed multi-tenant product. Keep public deployment in walkthrough mode until model evidence exists.

The public runtime is audited in CI with pip-audit. The larger ML environment is isolated from the public host; its exact package versions are recorded with each experiment. Request bodies are bounded at 24 KB, including chunked uploads.
