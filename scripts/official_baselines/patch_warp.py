"""The only change made to the official WARP code (third_party/xtr-warp): the optional OpenVINO runtime is imported at
module level although the default PyTorch runtime (used here) never touches it, and its dependency (optimum-intel)
pulls a newer torch. The two imports are guarded; nothing that runs is modified. Idempotent.

    python scripts/official_baselines/patch_warp.py third_party/xtr-warp
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "warp" / "engine" / "runtime" / "openvino_model.py"
src = path.read_text(encoding="utf-8")
old = "import openvino.runtime as ov\nfrom optimum.intel.openvino import OVModelForSeq2SeqLM\n"
new = ("try:   # optional runtime, unused with the default PyTorch runtime\n"
       "    import openvino.runtime as ov\n"
       "    from optimum.intel.openvino import OVModelForSeq2SeqLM\n"
       "except ImportError:\n"
       "    ov = OVModelForSeq2SeqLM = None\n")
if old in src:
    path.write_text(src.replace(old, new), encoding="utf-8")
    print("patched", path)
else:
    print("already patched" if new in src else "pattern not found", path)
(Path(sys.argv[1]) / "PATCHES.md").write_text(
    "Local change for the ReCo baseline runs: optional OpenVINO imports in warp/engine/runtime/openvino_model.py are\n"
    "guarded (scripts/official_baselines/patch_warp.py). The default PyTorch runtime, the indexer and the search kernels are unmodified.\n",
    encoding="utf-8")
