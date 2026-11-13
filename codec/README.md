# Concealment-aware layered 3DGS codec

PyTorch implementation of the codec of the paper: an anchor-based 3D Gaussian Splatting representation split into a
replicated base layer and an enhancement layer that is mapped to the parallel streams of the link, a factorized
entropy model, and a generative concealment network that predicts the enhancement layer of anchors whose stream failed.
Rendering uses the [gsplat](https://github.com/nerfstudio-project/gsplat) rasterizer.

## Pipeline

1. **Reference 3DGS** (`train_3dgs.py`): per scene, gsplat with MCMC densification capped at 500,000 Gaussians,
   30,000 steps, SH degree 3, L1 + 0.2 D-SSIM, opacity and scale regularisation 0.01. Every 8th image is a test view.
2. **Anchors** (`anchor_model.py`): the occupied voxels of the reference Gaussians with opacity > 0.1; the voxel size is
   chosen for a target number of anchors (250,000). Each anchor has a position (sent on the protected anchor channel),
   48 feature channels, 10 offsets and 6 log-scales; shared MLPs (hidden 32) produce 10 neural Gaussians per anchor
   (structure of Scaffold-GS).
3. **Layered codec with joint concealment** (`train_layered.py`):
   - base layer: first 12 feature channels + 6 scalings (18 values), replicated with the positions on the most
     reliable modes; enhancement layer: 36 feature channels + 30 offsets (66 values), packetised into tokens of 64
     Morton-consecutive anchors and mapped to the streams;
   - quantization-aware training (steps 0.05 / 0.02 / 0.02 for features / offsets / scalings) from step 10,000;
   - factorized entropy model (`RateModel`): a Laplace density per channel with learned location and scale; the code
     length of a quantized value q is -log2(F(q + D/2) - F(q - D/2)); optional rate penalty `--rate-lambda`;
   - concealment network trained jointly (`--aware 1`): every step renders the clean view and a view in which the
     enhancement layer of the anchors on K failed streams (M in {6, 8}, K = 1/2/3 with probability 0.7/0.2/0.1) is
     predicted by the network; loss = clean rendering loss + concealed rendering loss + 0.1 normalised attribute MSE;
     `--aware 0` is the separate design (concealment gradients reach neither the representation nor the decoder MLPs);
   - anchors that contribute to no training view are pruned at the end.
4. **Concealment networks** (`conceal_net.py`):
   - `LiteConcealNet` (used in the final design, about 254k parameters): neighbour tokens (16 nearest anchors:
     relative-position Fourier features, base layer, enhancement layer if received, received flag) encoded once by a
     two-layer MLP; the lost anchor's query (from its own base layer) is refined by two single-query cross-attention
     blocks (4 heads, width 96) with feed-forward blocks;
   - `EdgeConcealNet`: per-anchor encodings computed once when an anchor arrives and cached at the edge; the relative
     position enters only as an attention bias per head; `forward_table()` equals `forward()` exactly;
   - `LayeredConcealNet`: a 4-layer transformer over the 17 tokens, the reference network of the development stage;
   - `BitLinear` (ternary weights with absmean scale, 8-bit activations) for quantization-aware variants.
5. **Evaluation**:
   - `eval_layered.py`: every held-out test view, failure patterns drawn with their empirical frequencies from the
     1e6-slot traces of a configuration, four mapping policies, methods drop / base-only / inverse-distance
     interpolation / concealment network(s); PSNR, SSIM, LPIPS (VGG) against the ground truth and PSNR against the
     loss-free rendering;
   - `eval_trajectory.py`: camera path in capture order with an edge cache (optionally LRU-limited); temporally
     correlated outage slots;
   - `eval_frames.py`: frame-level scenario (11 slots per frame at 90 frames/s, edge cache, new anchors sent in the
     first slot of the frame) comparing no protection, concealment, in-frame retransmission and an (M, M - r) erasure
     code across the streams at the same redundancy, alone and combined with concealment; per-slot failure patterns of
     the correlated traces;
   - `bench_latency.py`: CUDA-event timing of neighbour gathering, concealment (FP32 / FP16 / compiled), per-anchor
     encoding, anchor-to-Gaussian decoding and rendering; `bench_clean.sh` pauses other GPU jobs during the measurement.

## Running

```bash
python fetch_mipnerf360.py --out data/360_v2 --scenes garden,stump
python train_3dgs.py --scene garden
python train_layered.py --scene garden --tag _v2a_r1 --anchors 250000 --feat-dim 48 --base-dim 12 \
       --net lite --d 96 --layers 2 --aware 1 --rate-lambda 5e-5
python eval_layered.py --scene garden --tag _v2a_r1 --cfg T8R8 --s2 2.5 --n-pat 6
python eval_frames.py --scene garden --tag _v2a_r1 --cfg T8R8 --tau 10 --n-anchor 4 --chains 10
python summarize.py            # one line per evaluation file
```
The physical-layer inputs are read from `phy/` (`modeset_<s2>.json`, `traces_<s2>.npz`, `traces_corr/`,
`patterns/`); copy or link them from `../phy_sim/results/`.

Scene folders: `data/360_v2/<scene>` (bicycle, garden, stump at `images_4`; room, counter, kitchen, bonsai at
`images_2`), `data/tandt/<truck|train>` and `data/db/<playroom|drjohnson>` at `images` (`scene_data.DATASETS`).

## Hardware and runtime

Trainings and evaluations ran on an NVIDIA L40S (48 GB) shared with other jobs, which lengthens the wall-clock times in
the logs by a factor of about 3 to 5; development runs used a GTX 1070. On an idle L40S the reference 3DGS takes about
5 minutes per outdoor scene. Latency numbers in `results/bench_l40s_idle.json` were taken with all other jobs of this
project paused.

## Results layout

See section 4 of the top-level README. `results/logs/` contains the training logs and the queue logs of the shared
server (`guard*.log`, `ram_watchdog.log`).
