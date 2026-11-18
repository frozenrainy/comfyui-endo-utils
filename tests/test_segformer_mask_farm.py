"""SegformerMask vs Easy-Use humanSegmentation's segformer path, on the farm (needs the model + a GPU).
  python tests/test_segformer_mask_farm.py <pack dir> <video> <classes>  → fails if IoU < 0.995
The full comparison/timing script lives in the workspace (tools/abtest/segmask_1118/cmp_segmask.py)."""
import sys, types, importlib.util, av, numpy as np, torch, torch.nn.functional as F
from PIL import Image
pack, video, classes = sys.argv[1], sys.argv[2], [int(c) for c in sys.argv[3].split(",")]
sys.modules["nodes"] = types.SimpleNamespace(LoraLoaderModelOnly=object)
sys.modules["folder_paths"] = types.SimpleNamespace(get_filename_list=lambda k: [], models_dir="/data/comfyfarm/model_2")
spec = importlib.util.spec_from_file_location("eu", pack + "/__init__.py"); eu = importlib.util.module_from_spec(spec); spec.loader.exec_module(eu)
MP = sys.modules["folder_paths"].models_dir + "/segformer_b3_clothes"
fr = np.stack([f.to_ndarray(format="rgb24") for f in av.open(video).decode(video=0)][:9])
img = torch.from_numpy(fr).float() / 255.0
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
proc = SegformerImageProcessor.from_pretrained(MP); ref_m = AutoModelForSemanticSegmentation.from_pretrained(MP).eval()
with torch.inference_mode():
    ref = np.stack([np.isin(F.interpolate(ref_m(**proc(images=Image.fromarray(np.clip(255. * im.numpy(), 0, 255).astype(np.uint8)),
                    return_tensors="pt")).logits, size=im.shape[:2], mode="bilinear", align_corners=False).argmax(1)[0].numpy(), classes) for im in img])
m = eu._segformer_load(MP).to("cuda")
out = eu._segformer_masks(img, m, classes, 8, "cuda").numpy() > 0.5
iou = (out & ref).sum() / max((out | ref).sum(), 1)
assert ref.any(), "reference mask empty — pick a class present in the video"
assert iou >= 0.995, iou
bad = eu._segformer_masks(img, m, [c + 1 for c in classes], 8, "cuda").numpy() > 0.5      # negative control: another class
assert (bad & ref).sum() / max((bad | ref).sum(), 1) < 0.5
print("ok IoU", round(float(iou), 5))
