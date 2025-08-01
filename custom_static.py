from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.requests import Request
import os

class CustomStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response: Response = await super().get_response(path, scope)
        if path.startswith("results") or path.startswith("preview"):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET"
            response.headers["Access-Control-Allow-Headers"] = "*"
        return response
