# ComfyUI node pack: run Strata (Qwen3.8-Flash-Next) as a graph node.
# only import if running as a custom node
try:
    import comfy.utils
except ImportError:
    pass
else:
    from .strata import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    print(
        "[Strata]: activated "
        f"{len(NODE_CLASS_MAPPINGS)} Strata nodes "
        "(start -> prompt -> stop, one instance at a time)."
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
