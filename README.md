# comfyui-endo-utils

ComfyUI nodes for depth-controlled video pipelines (WAN VACE and the like): depth maps a model can follow, clips cut
into parts that join without a seam, and anchor frames that stay put.

| Node | What it does |
|---|---|
| **DepthInverseNormalize** | Raw estimated depth → a VACE control: inverse depth (near = bright) normalised by one percentile pair over the **whole clip** — no flicker from per-frame normalisation, no sky eating the range |
| **DepthStats** | The percentiles of a clip, sampled once — the reference the chained normalisation uses |
| **DepthNormalizeChained** | The same normalisation for a clip processed in parts, every part on the clip's reference |
| **DepthPadExtend** | Pads depth by repeating its edges (outpainting) |
| **DepthPadDirectional** · **DepthPadSettings** | Padding per side (top, bottom, left, right; pixels, aspect kept) |
| **FrameContactSheet** | Frames of a video on one numbered grid, to pick an anchor frame by number |
| **AnchorPick** | Keeps the picked anchor as a file; with `lock` on it uses the kept file and does not ask for the candidate video at all, so the candidates are not generated again (generation differs from run to run even with the same seed) |
| **VideoLoadRetime** · **VideoRetimeSettings** | A video at a constant frame rate, from low, variable or duplicated-frame sources |
| **VideoClipInfo** | Scans a clip: frame count, load size, diagnostics |
| **VideoLoadFrames** | Loads a range of frames — a loop over the parts of a long clip |
| **FrameSeqWrite** | Writes a part's frames to files (no frames piling up in RAM) |
| **FrameSeqVideo** | Saves the video — from the written frame files (`lock` on) or from its input images |
| **SegformerMask** | Clothes/body-part masks from a segformer model (e.g. `segformer_b3_clothes`), batched on the GPU. `merge_alike` also takes the classes segformer mixes up with the chosen ones (upper clothes 4 ↔ dress 7, a dress's lower part as skirt 5); `temporal` joins each frame's mask with its neighbours' so a frame or two the labels miss is not left out |
| **SegformerMaskCheck** | Before anything is drawn: are the chosen classes found all through the video? On frames spread over the clip it reports the mask area, the clothes it does see, a sheet with the mask in red — and stops the run if too many frames have no mask (those frames would come out unchanged) |
| **MaskBBox** | One box around a batch of masks, to process only around the target and paste it back |

Tooltips and display names are in Korean.

## Install

Clone into `ComfyUI/custom_nodes/`. Nothing else to install: it uses numpy, torch, Pillow and PyAV, which ComfyUI
already requires. The segformer nodes also need `transformers` (most environments with segmentation nodes have it).

With ComfyFarm: *System → Rollout → Node packs → Add node pack*, `https://github.com/frozenrainy/comfyui-endo-utils`.

## License

GPL-3.0-or-later (it imports ComfyUI's own modules).
