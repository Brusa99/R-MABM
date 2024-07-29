from gymnasium import register

from .cats import Cats, CatsLog

register(
    id="Cats",
    entry_point="rmabm.environments:Cats",
)

register(
    id="CatsLog",
    entry_point="rmabm.environments:CatsLog",
)


