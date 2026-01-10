import random

from pydantic import BaseModel

from minisweagent import Model
from minisweagent.models import get_model


class RouletteModelConfig(BaseModel):
    model_kwargs: list[dict]
    """The models to choose from"""
    model_name: str = "roulette"


class RouletteModel:
    def __init__(self, *, config_class: type = RouletteModelConfig, **kwargs):
        """This "meta"-model randomly selects one of the models at every call"""
        self.config = config_class(**kwargs)
        self.models = [get_model(config=config) for config in self.config.model_kwargs]
        self._n_calls = 0

    def get_template_vars(self, **kwargs) -> dict:
        return self.config.model_dump()

    def select_model(self) -> Model:
        return random.choice(self.models)

    def query(self, *args, **kwargs) -> dict:
        model = self.select_model()
        self._n_calls += 1
        response = model.query(*args, **kwargs)
        response["model_name"] = model.config.model_name
        return response

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class InterleavingModelConfig(BaseModel):
    model_kwargs: list[dict]
    sequence: list[int] | None = None
    """If set to 0, 0, 1, we will return the first model 2 times, then the second model 1 time,
    then the first model again, etc."""
    model_name: str = "interleaving"


class InterleavingModel(RouletteModel):
    def __init__(self, *, config_class: type = InterleavingModelConfig, **kwargs):
        """This "meta"-model alternates between the models in the sequence for every call"""
        super().__init__(config_class=config_class, **kwargs)

    def select_model(self) -> Model:
        if self.config.sequence is None:
            i_model = self._n_calls % len(self.models)
        else:
            i_model = self.config.sequence[self._n_calls % len(self.config.sequence)]
        return self.models[i_model]
