from typing import Tuple

from rmabm.environments import Cats


class Dummy:
    """Dummy class that doesn't do anything.

    This class exists in order to obtain baseline rewards and other metrics.
    All class methods accept any arguments in order to be used interchangeably with other agents. The methods either
    don't do anything or return a fixed value, of a compatible type of the return of other agents classes.

    Args:
        agent_id: unique identifier of the agent, derived from the environment where the agent will be used.
        environment: environment used to deduce the action length. If None, action length will be 2.

    Attributes:
        action_length: length of the tuple returned by get_action. Deduced by the environment if provided otherwise 2.

    """

    def __init__(self, agent_id: int, environment: Cats | None = None):
        self.agent_id = agent_id

        # length of the tuple returned by get_action
        self.action_length: int = 2

        if environment:
            try:
                self.action_length = environment.action_space.shape[0]
            except TypeError:
                # action space is a dictionary
                self.action_length = len(environment.action_space.keys())


    @staticmethod
    def bin_obs(*args, **kwargs) -> tuple[int, int]:
        """Dummy method that doesn't do anything.

        The method returns a fixed value of (-1, -1) to be compatible with the return of other agents classes.
        (-1, -1) is not an expecteed value to be returned by the bin_state method of other agents classes. It is though
        a valid index for the Q table of the QLearner class.
        The index can be used when plotting the binned states as it will not interfere with the binned states used by
        other agents.

        Args:
            *args: arbitrary positional arguments.
            **kwargs: arbitrary keyword arguments.

        Returns:
            (-1, -1) dummy return value.

        """
        return -1, -1

    def get_action(self, *args, **kwargs) -> tuple[str, ...]:
        """Dummy method that doesn't do anything.

        The method always returns a string "dummy" that will be ignored by the environment.

        Args:
            *args: arbitrary positional arguments.
            **kwargs: arbitrary keyword arguments.

        Returns:
            "dummy" value that will be ignored by the environment.

        """
        return ("dummy",) * self.action_length

    @staticmethod
    def train(*args, **kwargs):
        """Dummy training method that doesn't do anything.

        Args:
            *args: arbitrary positional arguments.
            **kwargs: arbitrary keyword arguments.
        """

        pass
