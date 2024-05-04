class Dummy:
    """Dummy class that doesn't do anything.

    This class exists in order to obtain baseline rewards and other metrics.
    All class methods accept any arguments in order to be used interchangeably with other agents. The methods either
    don't do anything or return a fixed value, of a compatible type of the return of other agents classes.

    Args:
        agent_id: unique identifier of the agent, derived from the environment where the agent will be used.

    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    @staticmethod
    def bin_state(*args, **kwargs) -> tuple[int, int]:
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

    @staticmethod
    def get_action(*args, **kwargs) -> str:
        """Dummy method that doesn't do anything.

        The method always returns a string "dummy" that will be ignored by the environment.

        Args:
            *args: arbitrary positional arguments.
            **kwargs: arbitrary keyword arguments.

        Returns:
            "dummy" value that will be ignored by the environment.

        """
        return "dummy"

    @staticmethod
    def train(*args, **kwargs):
        """Dummy training method that doesn't do anything.

        Args:
            *args: arbitrary positional arguments.
            **kwargs: arbitrary keyword arguments.
        """

        pass
