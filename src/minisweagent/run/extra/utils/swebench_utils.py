from minisweagent import Environment
from minisweagent.environments import get_environment


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    image_name = get_swebench_docker_image_name(instance)
    env_config = config.setdefault("environment", {})
    if env_config.get("environment_class") == "singularity":
        image_name = "docker://" + image_name
    env_config["image"] = image_name
    return get_environment(env_config, default_type="docker")
