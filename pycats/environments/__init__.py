from gymnasium import register

from .cats import Cats

register(
    id="Cats",
    entry_point="pycats.environments:Cats",
)


