"""MaskBBox: union box over the batch, margin, multiple of 16, min size, kept inside the frame.
  python tests/test_mask_bbox_torch.py <pack dir>   (needs real torch)"""
import sys, types, importlib.util, torch
sys.modules["nodes"] = types.SimpleNamespace(LoraLoaderModelOnly=object)
sys.modules["folder_paths"] = types.SimpleNamespace(get_filename_list=lambda k: [])
spec = importlib.util.spec_from_file_location("eu", sys.argv[1] + "/__init__.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
N = m.MaskBBox()
H, W = 944, 560
mk = torch.zeros(3, H, W)
mk[0, 100:150, 200:260] = 1; mk[2, 400:420, 300:310] = 1             # union spans two frames
x, y, w, h = N.run(mk, True, 64, 16, 256)
assert w % 16 == 0 and h % 16 == 0 and w >= 256 and h >= 256, (x, y, w, h)
assert x <= 200 - 64 and x + w >= 310 + 64 and y <= 100 - 64 and y + h >= 420 + 64, (x, y, w, h)   # covers union + margin
assert 0 <= x and x + w <= W and 0 <= y and y + h <= H
mk2 = torch.zeros(1, H, W); mk2[0, 0:10, W - 10:W] = 1                   # corner: pushed inside the frame
x, y, w, h = N.run(mk2, True, 64, 16, 256)
assert (x + w, y) == (W, 0) and w == 256 and h == 256, (x, y, w, h)
mk3 = torch.zeros(2, H, W); mk3[:, 10:930, 5:555] = 1                     # bigger than frame after margin: whole frame
assert N.run(mk3, True, 64, 16, 256) == (0, 0, W, H)
assert N.run(torch.zeros(2, H, W), True, 64, 16, 256) == (0, 0, W, H)    # nothing found: whole frame
assert N.run(mk, False, 64, 16, 256) == (0, 0, W, H)                     # off: whole frame
print("ok", m.NODE_DISPLAY_NAME_MAPPINGS["MaskBBox"])
