from .inpainting_module import Inpainter

try:                                     # PyTorch is optional
    from .multi_stream_net import MultiStreamForensicNet, build_inputs, TORCH_OK
except Exception:                        # pragma: no cover
    MultiStreamForensicNet, build_inputs, TORCH_OK = None, None, False

__all__ = ["Inpainter", "MultiStreamForensicNet", "build_inputs", "TORCH_OK"]
