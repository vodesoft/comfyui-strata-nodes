# ComfyUI node pack: run Strata (Qwen3.8-Flash-Next) as a graph node.
from .strata import StrataExtension

__all__ = ["StrataExtension"]


async def comfy_entrypoint():
    return StrataExtension()
