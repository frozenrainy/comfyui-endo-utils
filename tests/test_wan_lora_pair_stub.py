import sys, types, importlib.util
calls = []
for _m in ("numpy", "torch"): sys.modules.setdefault(_m, types.ModuleType(_m))
class FakeLoader:
    def load_lora_model_only(self, model, name, s): calls.append((model, name, s)); return (f"{model}+{name}@{s}",)
sys.modules["nodes"] = types.SimpleNamespace(LoraLoaderModelOnly=FakeLoader)
sys.modules["folder_paths"] = types.SimpleNamespace(get_filename_list=lambda k: ["a_HIGH.safetensors", "a_LOW.safetensors"])
spec = importlib.util.spec_from_file_location("eu", sys.argv[1] + "/__init__.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
P = m.WanLoraPair
it = P.INPUT_TYPES()["required"]
assert it["lora_high"][0][0] == "None" and "a_HIGH.safetensors" in it["lora_high"][0]
n = P()
assert n.run("H", "L", "a_HIGH.safetensors", 0.8, "a_LOW.safetensors", 0.6) == ("H+a_HIGH.safetensors@0.8", "L+a_LOW.safetensors@0.6")
assert calls == [("H", "a_HIGH.safetensors", 0.8), ("L", "a_LOW.safetensors", 0.6)]
calls.clear()
assert n.run("H", "L", "None", 1.0, "a_LOW.safetensors", 0.0) == ("H", "L") and calls == [], calls     # off = untouched, not read
assert n.run("H", "L", "a_HIGH.safetensors", 1.0, "None", 1.0) == ("H+a_HIGH.safetensors@1.0", "L")
assert n._h is not n._l
print("ok", m.NODE_DISPLAY_NAME_MAPPINGS["WanLoraPair"])
