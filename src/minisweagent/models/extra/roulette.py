import random
from dataclasses import asdict, dataclass

from minisweagent import Model
from minisweagent.models import get_model


@dataclass
class RouletteModelConfig:
    model_kwargs: list[dict]
    """The models to choose from"""
    model_name: str = "roulette"


class RouletteModel:
    def __init__(self, *, config_class: type = RouletteModelConfig, **kwargs):
        """This "meta"-model randomly selects one of the models at every call"""
        self.config = config_class(**kwargs)
        self.models = [get_model(config=config) for config in self.config.model_kwargs]

    @property
    def cost(self) -> float:
        return sum(model.cost for model in self.models)

    @property
    def n_calls(self) -> int:
        return sum(model.n_calls for model in self.models)

    def get_template_vars(self) -> dict:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def select_model(self) -> Model:
        return random.choice(self.models)

    def query(self, *args, **kwargs) -> dict:
        model = self.select_model()
        response = model.query(*args, **kwargs)
        response["model_name"] = model.config.model_name
        return response


@dataclass
class InterleavingModelConfig:
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
            i_model = self.n_calls % len(self.models)
        else:
            i_model = self.config.sequence[self.n_calls % len(self.config.sequence)]
        return self.models[i_model]
