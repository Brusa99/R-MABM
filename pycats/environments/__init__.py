from gymnasium import register

from .cats import Cats, CatsLog

register(
    id="Cats",
    entry_point="pycats.environments:Cats",
)

register(
    id="CatsLog",
    entry_point="pycats.environments:CatsLog",
)


