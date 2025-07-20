from typing import Union

from fastapi import FastAPI
from chainlit.utils import mount_chainlit

app = FastAPI()

mount_chainlit(app=app, target="cl_app.py", path="/")