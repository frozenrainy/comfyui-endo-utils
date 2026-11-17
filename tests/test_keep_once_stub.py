"""KeepOnce: the input is requested only the first time within one prompt; a new prompt starts afresh.
  python3 tests/test_keep_once_stub.py <pack dir>"""
import sys, types, importlib.util
for m in ("numpy", "torch"): sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["nodes"] = types.SimpleNamespace(LoraLoaderModelOnly=object)
sys.modules["folder_paths"] = types.SimpleNamespace(get_filename_list=lambda k: [])
spec = importlib.util.spec_from_file_location("eu", sys.argv[1] + "/__init__.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
K = m.KeepOnce()
class DP: pass
p1, p2 = DP(), DP()
calls = []
def execute(dp, uid, producer):
    """What the executor does: ask which lazy inputs are needed, evaluate them, run."""
    need = K.check_lazy_status(value=None, dynprompt=dp, unique_id=uid)
    value = producer() if "value" in need else None
    return K.run(value=value, dynprompt=dp, unique_id=uid)[0]
prod = lambda tag: (lambda: calls.append(tag) or f"result-{tag}")
assert execute(p1, "7", prod("a")) == "result-a"
assert execute(p1, "7", prod("b")) == "result-a"            # same prompt, same node: producer not run again
assert execute(p1, "8", prod("c")) == "result-c"            # another KeepOnce node keeps its own value
assert execute(p2, "7", prod("d")) == "result-d"            # a new prompt starts afresh
assert calls == ["a", "c", "d"], calls
assert not (m.ANY != "MODEL") and not (m.ANY != "IMAGE")
print("ok", m.NODE_DISPLAY_NAME_MAPPINGS["KeepOnce"])
