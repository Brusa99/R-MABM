from gymnasium import register

from .cats import Cats

register(
    id="Cats-v0",
    entry_point="pycats.environments:Cats",
)


