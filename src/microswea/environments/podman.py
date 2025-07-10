from dataclasses import dataclass

from microswea.environments.docker import DockerEnvironment, DockerEnvironmentConfig


@dataclass
class PodmanEnvironmentConfig(DockerEnvironmentConfig):
    executable: str = "podman"


class PodmanEnvironment(DockerEnvironment):
    def __init__(self, config_class: type = PodmanEnvironmentConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
