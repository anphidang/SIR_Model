# SIR reproduction and node-representation study (fork)

This is **not** the official SIR repository. It is a fork of
[intuitive-robots/SIR_Model](https://github.com/intuitive-robots/SIR_Model), created for a project in the
IROBMAN Project Lab (*Praktikum zur intelligenten Robotermanipulation*, TU Darmstadt, summer semester 2026).
It adds what is needed to train the sparsified SIR configuration, together with the experiments described
in the accompanying report:

**Report:** *Structured Image Representations for Explainable Robot Learning — a reproduction of SIR and an
assessment of whether its sparsification transfers across node modalities* (An-Phi Dang, 2026)

All credit for the method and the original code goes to the SIR authors:

> Paul Mattes, Jan Schwab, Jens Bosch, Maximilian Li, Nils Blank, Minh-Trung Tang, Moritz Haberland and
> Rudolf Lioutikov. **SIR: Structured Image Representations for Explainable Robot Learning.** CVPR 2026.
> [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Mattes_SIR_Structured_Image_Representations_for_Explainable_Robot_Learning_CVPR_2026_paper.html) ·
> [Project page](https://intuitive-robots.github.io/SIR_website/) ·
> [Original code](https://github.com/intuitive-robots/SIR_Model)

If you use the method, please cite the original paper (see [Citation](#citation)).

## What this fork changes

The fork is based on upstream commit `5ea851e`; upstream `19d1d57` and `c0e8a2e` differ from it only in the README.

| Change | Files | Purpose |
|:---|:---|:---|
| `Multi_XAI_GNN` wrapper | `networks/graph_encoder/gnn.py` | The released `xai_gnn.yaml` points to a class `Multi_XAI_GNN` that does not exist upstream, so the sparsified configuration cannot be trained there. The wrapper runs the released `Sparsification_Module` per modality and passes the retained sub-graph to the GNN. The module itself only got small fixes; its hyperparameters are unchanged. |
| Score-weighted readout, `selection_accuracy` logging | `networks/graph_encoder/gnn.py` | Pooling weighted by node scores; logging-only diagnostic of which nodes are retained |
| Unit tests for the sparsifier | `test_xai_gnn.py` | CPU-only checks on synthetic graphs (shapes, *k* nodes kept, gradient flow) |
| 3D oriented bounding boxes (`bb3d_coordinates`) | `utils/generate_3d_bb_dataset_robocasa.py`, `utils/generate_3d_bb_graph_dataset_robocasa.py`, `dataloader/`, `manager/` | Node geometry as a 12-D oriented 3D box relative to the gripper (`full` or `translation_only` frame) |
| CLIP label embeddings (`clip_labels`) | `utils/generate_graph_dataset_robocasa.py` | Node labels as L2-normalised CLIP ViT-B/32 text embeddings instead of one-hot vectors |
| Node masking (`mask_objects`) | `dataloader/utils.py`, `manager/robocasa_manager.py`, `envs/robocasa/kitchen.py` | Remove given object classes from every graph, at training and rollout time. The mask is applied at load time to the full graphs and is not part of the cache key |
| Data-pipeline and RoboCasa fixes | `dataloader/`, `manager/`, `envs/robocasa/kitchen.py` | Fixes needed to run the pipeline end to end |
| Sub-graph inspection | `utils/inspect_subgraphs.py` | Runs a trained sparsifier over all demonstration frames, records the retained nodes and the exact per-frame chance level |
| Image tensors | `utils/make_img_tensors.py` | Rebuilds the per-demo image tensors the data loader expects for the image-only baseline |
| Run-time accounting | `utils/run_time.py` | Wall-clock run durations from `main.log`, excluding suspend periods |
| Run scripts | `run_all.sh`, `configs/mask_sweep.yaml` | SIR / fully connected / image runs on TurnOffSinkFaucet; masking sweep |

Note that `configs/method/diffusion.yaml` now uses the sparsified encoder (`xai_gnn`) by default. Use
`method/graph_encoder=gnn` for the fully connected graph.

## Installation

SIR, robosuite and RoboCasa must be cloned **next to each other** in one parent folder, not inside each other.

1. Clone this fork and create the conda environment (the script asks for an environment name):

   ```bash
   git clone -b sir-anpassungen https://github.com/anphidang/SIR_Model.git
   cd SIR_Model
   bash install.sh
   conda activate <your-env-name>
   ```

2. Install robosuite from the fork [anphidang/robosuite](https://github.com/anphidang/robosuite),
   branch `robocasa_v0.1`. It is upstream robosuite `robocasa_v0.1` (commit `8ea59ea0`) with the two changes
   SIR requires already applied: `IMAGE_CONVENTION = "opencv"` in `robosuite/macros.py`, and skipping of
   stale contact/visual geoms in `Task.merge_objects` (`robosuite/models/tasks/task.py`):

   ```bash
   cd ..
   git clone -b robocasa_v0.1 https://github.com/anphidang/robosuite.git
   cd robosuite
   pip install -e .
   python robosuite/scripts/setup_macros.py
   ```

   `setup_macros.py` copies `macros.py` to `macros_private.py`; if a `macros_private.py` already exists,
   check that it also sets `IMAGE_CONVENTION = "opencv"`. Windows users: see the
   [robosuite installation guide](https://robosuite.ai/docs/installation.html).

3. Install RoboCasa from the fork [anphidang/robocasa-sir](https://github.com/anphidang/robocasa-sir),
   branch `sir-anpassungen`. It is upstream RoboCasa at commit `370f986` with the changes to `kitchen.py`
   that SIR requires (seeding, `camera_segmentations="class"`, `mujoco_objects`) already applied:

   ```bash
   cd ..
   git clone -b sir-anpassungen https://github.com/anphidang/robocasa-sir.git robocasa
   cd robocasa
   pip install -e .
   python robocasa/scripts/download_kitchen_assets.py
   python robocasa/scripts/setup_macros.py
   ```

4. Adjust the local paths and the W&B account, which are currently set to the author's machine:

   | File | Key |
   |:---|:---|
   | `configs/manager/robocasa.yaml` | `data_path` |
   | `configs/main.yaml` | `trainer.log_dir`, `wandb.entity`, `wandb.project` |

   These can also be overridden on the command line, e.g. `manager.data_path=/path/to/data`.

## Data

The pre-built graph datasets from the SIR authors are on HuggingFace:
[MrLayen/SIR_robocasa](https://huggingface.co/datasets/MrLayen/SIR_robocasa). Place them under `data_path`.

The 3D bounding boxes are not part of that dataset and must be generated per task before training with
`bb3d_coordinates`. The `--frame_mode` must match `manager.bb3d_frame_mode`:

```bash
python utils/generate_3d_bb_dataset_robocasa.py --task CloseSingleDoor --frame_mode translation_only
```

## Running the experiments

All runs in the report use CloseSingleDoor, seed 42 (43 for the second SIR seed), 50 epochs and an
evaluation with 100 rollouts (`trainer.eval_n_times=20` × `manager.times_repeat=5`). A common prefix:

```bash
COMMON="manager.task_names=[CloseSingleDoor] trainer.seed=42 trainer.epochs=50 trainer.test_bool=True trainer.eval_n_times=20"
```

| Name in the report | W&B run name | Overrides (in addition to `$COMMON`) |
|:---|:---|:---|
| FC (no label) | R_fc_graph | `method/graph_encoder=gnn manager.graph_modalities=[bb_coordinates,cropped_image_feature]` |
| SIR baseline | E1b | `method/graph_encoder=xai_gnn manager.graph_modalities=[bb_coordinates,cropped_image_feature]` |
| SIR, seed 43 | R_sir_seed43 | as SIR baseline with `trainer.seed=43` |
| SIR masked | E1a | SIR baseline + `manager.mask_objects=[PandaMobile,PandaGripper,Wall,Counter,Floor]` |
| SIR fixtures-masked | E1c | SIR baseline + `manager.mask_objects=[Wall,Counter,Floor]` |
| SIR + proprio | E1e | SIR baseline + `manager.prop_modalities=[robot0_base_to_eef_pos,robot0_base_to_eef_quat,robot0_gripper_qpos]` |
| SIR masked + proprio | E1d | SIR masked + the same `manager.prop_modalities` |
| SIR one-hot | E2b | SIR baseline with `manager.graph_modalities=[bb_coordinates,cropped_image_feature,one_hot_labels]` |
| SIR-CLIP | E2a | SIR baseline with `manager.graph_modalities=[bb_coordinates,cropped_image_feature,clip_labels]` |
| FC-CLIP | E2fc | SIR-CLIP with `method/graph_encoder=gnn` |
| TopK-2D | E3b | SIR baseline + `method.graph_encoder.sparsification_layer.sampling_strategy=topk` |
| TopK-3D (SE(3)) | E3a-full | TopK-2D with `manager.graph_modalities=[bb3d_coordinates,cropped_image_feature] manager.bb3d_frame_mode=full` |
| TopK-3D (translation) | E3a-trans | as TopK-3D (SE(3)) with `manager.bb3d_frame_mode=translation_only` |

Example:

```bash
python main.py $COMMON method/graph_encoder=xai_gnn \
  manager.graph_modalities=[bb_coordinates,cropped_image_feature,clip_labels] \
  wandb.run_name=E2a
```

The unit tests for the sparsifier run without dataset or simulator:

```bash
python test_xai_gnn.py
```

## Known limitations

- Edge features are a constant 1.0 in the released code (`calculate_weight_dim_distance` in
  `dataloader/utils.py` is commented out); this fork does not change that.
- The image-only baseline (`manager.graph_modalities=[]`) did not produce a usable result on the
  machines used for the report.

## Citation

Please cite the original paper:

```bibtex
@InProceedings{Mattes_2026_CVPR,
    author    = {Mattes, Paul and Schwab, Jan and Bosch, Jens and Li, Maximilian Xiling and Blank, Nils and Tang, Minh-Trung and Haberland, Moritz and Lioutikov, Rudolf},
    title     = {SIR: Structured Image Representations for Explainable Robot Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {42484-42493}
}
```
