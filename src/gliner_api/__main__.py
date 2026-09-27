import uvicorn
from gliner_api.settings import config


def main() -> None:
    uvicorn.run("gliner_api.app:app", host=config.HOST, port=config.PORT, reload=True)


if __name__ == "__main__":
    main()
