"""SegformerMask merge_alike/temporal + SegformerMaskCheck, on the farm box (needs the segformer model; runs on CPU).
  python tests/test_segformer_check_farm.py <pack dir> <video:classes:merge:expect> ...
  expect = pass | fail  (e.g. "input/crop_top_skirt.mp4:7:0:fail")
Each video is sampled the way ④ samples it (48 frames spread over the clip, short side 560)."""
import sys, types, importlib.util, av, numpy as np, torch, torch.nn.functional as F
pack, cases = sys.argv[1], sys.argv[2:]
sys.modules["nodes"] = types.SimpleNamespace(LoraLoaderModelOnly=object)
sys.modules["folder_paths"] = types.SimpleNamespace(get_filename_list=lambda k: [], models_dir="/data/comfyfarm/model_2")
mm = types.SimpleNamespace(get_torch_device=lambda: torch.device("cpu"), soft_empty_cache=lambda: None)
sys.modules["comfy"] = types.SimpleNamespace(model_management=mm); sys.modules["comfy.model_management"] = mm
spec = importlib.util.spec_from_file_location("eu", pack + "/__init__.py"); eu = importlib.util.module_from_spec(spec); spec.loader.exec_module(eu)

# --- classes ---
assert eu._seg_classes("7") == [7]
assert eu._seg_classes("7", True) == [4, 5, 7]
assert eu._seg_classes("4", True) == [4, 7]
assert eu._seg_classes(" 1, 4 ", True) == [1, 4, 7]
assert eu._seg_classes("1,16", True) == [1, 16]
try:
    eu._seg_classes(" , ")
    raise AssertionError("empty classes accepted")
except ValueError:
    pass

# --- temporal union ---
m = torch.zeros(10, 4, 4); m[:, 1, 1] = 1; m[4, 1, 1] = 0; m[5, 1, 1] = 0          # 2-frame gap
m[0, 3, 3] = 1                                                                   # a blip at the start
assert torch.equal(eu._temporal_union(m, 0), m)
u = eu._temporal_union(m, 2)
assert u[:, 1, 1].min() == 1, "gap of 2 not filled with radius 2"
assert u[:3, 3, 3].tolist() == [1, 1, 1] and u[3:, 3, 3].sum() == 0, "blip spread wrong"
g = torch.zeros(12, 2, 2); g[0, 0, 0] = 1; g[11, 0, 0] = 1                          # negative control: 10-frame gap stays a gap
assert eu._temporal_union(g, 2)[:, 0, 0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
assert eu._temporal_union(m[:1], 3).shape == (1, 4, 4)

# --- check on real videos ---
def sample(path, n=48, short=560):
    fr = [f.to_ndarray(format="rgb24") for f in av.open(path).decode(video=0)]
    idx = np.unique(np.linspace(0, len(fr) - 1, min(n, len(fr))).round().astype(int))
    t = torch.from_numpy(np.stack([fr[i] for i in idx])).float() / 255
    h, w = t.shape[1:3]; s = short / min(h, w)
    H, W = int(round(h * s / 16) * 16), int(round(w * s / 16) * 16)
    return F.interpolate(t.movedim(-1, 1), size=(H, W), mode="bilinear", antialias=True).movedim(1, -1).clamp(0, 1)

node = eu.SegformerMaskCheck()
for case in cases:
    path, classes, merge, expect = case.rsplit(":", 3)
    img = sample(path)
    args = dict(image=img, model="segformer_b3_clothes", classes=classes, merge_alike=merge == "1", min_ratio=0.25, max_bad=0.10)
    sheet, report, ti, tv = node.run(stop=False, through_image=img[:1], through_value=7, **args)
    assert sheet.ndim == 4 and sheet.shape[0] == 1 and sheet.shape[-1] == 3
    assert ti.shape == img[:1].shape and tv == 7, "through_* not passed through"
    passed = report.startswith("✅")
    print(f"--- {path.split('/')[-1]} classes={classes} merge={merge} → {'pass' if passed else 'fail'} (expect {expect})\n{report}")
    assert passed == (expect == "pass"), case
    if not passed:                                                               # stop=True must raise with the same report
        try:
            node.run(stop=True, **args)
            raise AssertionError("stop=True did not stop")
        except ValueError as e:
            assert "잡힌 옷" in str(e) and "🎯 바꿀·지울 대상" in str(e)
print("ok")
