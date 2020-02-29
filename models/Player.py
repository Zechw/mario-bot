from abc import ABC

class Player(ABC):
    """
    Base class to handle game interactions
    """

    @abstractmethod
    def take_action(self, observation):
        pass
