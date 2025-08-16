import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

from minisweagent import Model
from minisweagent.models import get_model


@dataclass
class RouletteModelConfig:
    model_kwargs: list[dict]
    model_name: str ="roulette"
    developer_names: list[str] | None = None


class RouletteModel:
    def __init__(self, *, config_class: Callable = RouletteModelConfig, **kwargs):
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
        if self.config.developer_names:
            name = random.choice(self.config.developer_names)
            response["content"] = f"{name}:\n{response['content']}"
        return response
