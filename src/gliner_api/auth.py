from fastapi import Header, HTTPException, Request

from gliner_api.settings import config

USE_API_KEY = config.USE_API_KEY
API_KEY = config.API_KEY


def bearer_auth_dependency(
    request: Request,
    authorization: str | None = Header(default=None),
):
    if request.url.path == "/":
        return

    if not USE_API_KEY:
        return

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")

    token = authorization.split(" ", 1)[1]

    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")
