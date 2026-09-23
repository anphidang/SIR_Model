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
| `Multi_XAI_GNN` wrapper | `networks/graph_encoder/gnn.py` | The released `xai_gnn.yaml` points to a class `Multi_XAI_GNN` that does not exist upstream, so the sparsified configuration cannot be trained there. The wrapper runs the released `Sparsification_Module` per modality and passes the retained sub-graph to the GNN. Sparsifier hyperparameters are unchanged. |
| Score-weighted readout, `selection_accuracy` logging | `networks/graph_encoder/gnn.py` | Pooling weighted by node scores; logging-only diagnostic of which nodes are retained |
| Unit tests for the sparsifier | `test_xai_gnn.py` | CPU-only checks on synthetic graphs (shapes, *k* nodes kept, gradient flow) |
| 3D oriented bounding boxes (`bb3d_coordinates`) | `utils/generate_3d_bb_dataset_robocasa.py`, `utils/generate_3d_bb_graph_dataset_robocasa.py`, `dataloader/`, `manager/` | Node geometry as a 12-D oriented 3D box relative to the gripper (`full` or `translation_only` frame) |
| CLIP label embeddings (`clip_labels`) | `utils/generate_graph_dataset_robocasa.py` | Node labels as L2-normalised CLIP ViT-B/32 text embeddings instead of one-hot vectors |
| Node masking (`mask_objects`) | `dataloader/utils.py`, `manager/robocasa_manager.py`, `envs/robocasa/kitchen.py` | Remove given object classes from every graph, at training and rollout time |
| Data-pipeline and RoboCasa fixes | `dataloader/`, `manager/`, `envs/robocasa/kitchen.py` | Fixes needed to run the pipeline end to end |
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

| Report ID | Description | Overrides (in addition to `$COMMON`) |
|:---|:---|:---|
| R_fc_graph | Fully connected graph | `method/graph_encoder=gnn manager.graph_modalities=[bb_coordinates,cropped_image_feature]` |
| R_sir_seed42 / E1b | SIR | `method/graph_encoder=xai_gnn manager.graph_modalities=[bb_coordinates,cropped_image_feature]` |
| R_sir_seed43 | SIR, second seed | as above with `trainer.seed=43` |
| E1a | Robot and fixture nodes masked | SIR + `manager.mask_objects=[PandaMobile,PandaGripper,Wall,Counter,Floor]` |
| E1c | Fixture nodes masked | SIR + `manager.mask_objects=[Wall,Counter,Floor]` |
| E1d | E1a + proprioception | E1a + `manager.prop_modalities=[robot0_base_to_eef_pos,robot0_base_to_eef_quat,robot0_gripper_qpos]` |
| E1e | SIR + proprioception | SIR + the same `manager.prop_modalities` |
| E2a | CLIP labels | SIR with `manager.graph_modalities=[bb_coordinates,cropped_image_feature,clip_labels]` |
| E2b | One-hot labels | SIR with `manager.graph_modalities=[bb_coordinates,cropped_image_feature,one_hot_labels]` |
| E2fc | CLIP labels, fully connected | E2a with `method/graph_encoder=gnn` |
| E3a-full / E3a-trans | 3D boxes | `manager.graph_modalities=[bb3d_coordinates,cropped_image_feature] manager.bb3d_frame_mode=full` (or `translation_only`) + `method.graph_encoder.sparsification_layer.sampling_strategy=topk` |
| E3b | 2D control for E3 | as E3 with `bb_coordinates` instead of `bb3d_coordinates` |

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
