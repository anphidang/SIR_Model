import os
import re

import clip
import h5py
import networkx as nx
import numpy as np
import torch
from einops import einops
from tqdm import tqdm

from networks.vision_encoder.cnn import SimpleImageEncoder
from networks.vision_encoder.utils import crop_and_resize_to_64, crop_and_resize_to_64_for_fusion
from utils.create_graphs import create_graph_datapoint

OBJECT_NAMES_IMAGES = ['OmronMobileBase',
                'PandaMobile',
                'PandaGripper',
                'Wall',
                'Floor',
                'WallAccessory',
                'Counter', #6
                'CoffeeMachine',
                'Toaster',
                'Box',
                'HingeCabinet',
                'Sink', # 11
                'SingleCabinet',
                'Drawer',
                'FramedWindow',
                'Accessory', # 15
                'Stove',
                'Microwave',
                'Oven',
                'Fridge',
                'Dishwasher',
                'Stovetop',
                'Hood', # 22
                'Stool', # 23
                'obj', # 24
                'door_obj', # 25
                'drawer_obj',
                'obj_container',
                'distr_counter_0',
                'distr_counter_1',
                'distr_counter_2',
                'distr_counter_3',
                'distr_counter',
                'distr_cab',
                'distr_sink',
                'container',
                'cookware']

CLIP_LABEL_DIM = 512  # CLIP ViT-B/32 text embedding dimension

_clip_label_embeddings = None  # lazy cache, keyed by raw OBJECT_NAMES_IMAGES entry

def _object_name_to_phrase(name: str) -> str:
    """'PandaGripper' -> 'panda gripper', 'distr_counter_0' -> 'distr counter 0'."""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return spaced.replace("_", " ").lower().strip()

@torch.no_grad()
def get_clip_label_embeddings(device: str = "cpu") -> dict:
    """CLIP ViT-B/32 text embedding per name in OBJECT_NAMES_IMAGES, computed once and
    cached module-wide. Keyed by the raw object name, same lookup key one_hot_labels
    uses (OBJECT_NAMES_IMAGES.index(name)), so callers can swap the two modalities
    without touching the node-name bookkeeping."""
    global _clip_label_embeddings
    if _clip_label_embeddings is None:
        model, _ = clip.load("ViT-B/32", device=device)
        phrases = [_object_name_to_phrase(name) for name in OBJECT_NAMES_IMAGES]
        tokens = clip.tokenize(phrases).to(device)
        embeddings = model.encode_text(tokens).float()
        # Raw encode_text output has L2 norm ~8.7-11.7 (measured), vs. one-hot's fixed norm of
        # 1.0 - normalize so the label sub-vector isn't larger in scale purely from being CLIP.
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        _clip_label_embeddings = {
            name: embeddings[i].cpu() for i, name in enumerate(OBJECT_NAMES_IMAGES)
        }
    return _clip_label_embeddings

_random_label_embeddings = None

def get_random_label_embeddings(seed: int = 0) -> dict:
    """Capacity-matched control for clip_labels: one fixed random unit-norm CLIP_LABEL_DIM-d
    vector per name in OBJECT_NAMES_IMAGES. Same width and L2 norm (1.0) as the normalized CLIP
    text embeddings, so graph_encoder gets the identical input_dim, but no semantic structure
    and no relationship to the instruction embedding. Keyed by raw object name like the
    other label modalities."""
    global _random_label_embeddings
    if _random_label_embeddings is None:
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(len(OBJECT_NAMES_IMAGES), CLIP_LABEL_DIM, generator=gen)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        _random_label_embeddings = {name: emb[i] for i, name in enumerate(OBJECT_NAMES_IMAGES)}
    return _random_label_embeddings

@torch.no_grad()
def create_graphs_and_save(dataset_path: str,
                           task_name: str,
                           graph_modality: str,
                           encoder_model: torch.nn.Module = None,
                           mask_objects: list = None,
                        ):
    full_dataset_path = os.path.join(dataset_path, task_name, "dataset_raw.hdf5")
    
    if os.path.isfile(full_dataset_path):
        print(dataset_path + ": start hdf5 loading...")
        dataset = h5py.File(full_dataset_path, "r")
        print("hdf5 loaded!")
    else:
        raise Exception(f"HDF5 file in {dataset_path} is missing")
            
    all_data_left = []
    all_data_right = []
    
    raw_list = np.array(dataset)
    permutation = np.argsort(np.array([int(i.split('_', 1)[-1]) for i in raw_list]))

    start_zero = False
    if raw_list[permutation][0] == "demo_0":
        start_zero = True

    for key in tqdm(raw_list[permutation]):
        if start_zero: # some demonstrations start with 0
            demo_num = int(key.split("_")[-1])
        else: # human demonstrations start with 1
            demo_num = int(key.split("_")[-1]) - 1

        if demo_num < 0:
            print("WARNING: The demonstration index is not correctly starting with 0")
    
        left_graph: nx.Graph = nx.DiGraph()
        right_graph: nx.Graph = nx.DiGraph()
        
        temp_data_left = []
        temp_data_right = []

        img_tensor_all: torch.Tensor = torch.load(dataset_path + task_name + "/img_tensor_demo_" + str(demo_num) + ".pth")
        img_tensor_all: torch.Tensor = einops.rearrange(img_tensor_all, "l (num c h w) -> l num c h w", num=3, c=3, h=128)
        left_image: torch.Tensor = img_tensor_all[:, 1, :, :, :]
        right_image: torch.Tensor = img_tensor_all[:, 2, :, :, :]

        inner_raw_list = np.array(dataset[key])
        inner_permutation = np.argsort(np.array([int(i) for i in inner_raw_list]))
        
        for j in tqdm(range(int(list(inner_raw_list[inner_permutation])[-1]) + 1)):
            left_object_names, left_objects, right_object_names, right_objects = extract_graph_objects(dataset, key, j, graph_modality, left_image[j], right_image[j], encoder_model, mask_objects)
                
            left_data_point = create_graph_datapoint(left_graph, left_object_names, left_objects)
            temp_data_left.append(left_data_point)
            right_data_point = create_graph_datapoint(right_graph, right_object_names, right_objects)
            temp_data_right.append(right_data_point)
        
        all_data_left.append(temp_data_left)
        all_data_right.append(temp_data_right)

    if encoder_model is not None:
        if isinstance(encoder_model, SimpleImageEncoder):
            graph_modality = graph_modality + "_fusion_" + str(encoder_model.fc.out_features)
        else:
            graph_modality = graph_modality + "_" + str(encoder_model.fc_layer.out_features)
    if mask_objects:
        graph_modality = graph_modality + "_mask_" + "_".join(sorted(mask_objects))

    torch.save(all_data_left, dataset_path + task_name + "/" + graph_modality + "_left_image.pth")
    torch.save(all_data_right, dataset_path + task_name + "/" + graph_modality + "_right_image.pth")

def extract_graph_objects(dataset, key, j, graph_modality: str, left_img, right_img, encoder_model: torch.nn.Module = None, mask_objects: list = None):
    left_object_names: list = list(dataset[key][str(j)]["left_image"].keys())
    left_objects = dataset[key][str(j)]["left_image"]
    right_object_names: list = list(dataset[key][str(j)]["right_image"].keys())
    right_objects = dataset[key][str(j)]["right_image"]

    if graph_modality == "one_hot_labels":
        object_representation = "bb" # Just a placeholder, because one hot labels do not have extra representation
    elif graph_modality in ("clip_labels", "random_labels"):
        object_representation = "bb" # same placeholder reasoning as one_hot_labels
    elif graph_modality == "bb_coordinates":
        object_representation = "bb"
    elif graph_modality == "cropped_image_feature":
        object_representation = "MorphMask"

    left_objects, left_object_names = get_adjusted_objects_and_names(left_object_names, left_objects, object_representation, mask_objects)
    right_objects, right_object_names = get_adjusted_objects_and_names(right_object_names, right_objects, object_representation, mask_objects)
    
    left_objects, right_objects = get_node_features(left_objects, right_objects, left_object_names, right_object_names, left_img, right_img, graph_modality, encoder_model)
    
    left_objects = torch.stack(left_objects)
    right_objects = torch.stack(right_objects)
    
    return left_object_names, left_objects, right_object_names, right_objects

def get_adjusted_objects_and_names(object_names, objects, object_representation: str, mask_objects: list = None):
    mask_objects = mask_objects or []
    new_objects = []
    new_object_names = []
    for _, object_name in enumerate(object_names):
        if object_representation in object_name:
            temp = object_name.split("_" + object_representation)[0]
            if temp in mask_objects:
                continue
            new_object_names.append(temp)
            new_objects.append(objects[object_name][()])
    return new_objects, new_object_names

def get_node_features(objects_left,
                      objects_right,
                      object_names_left,
                      object_names_right,
                      img_left,
                      img_right,
                      graph_modality: str, 
                      encoder_model: torch.nn.Module = None,
                    ):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    
    left_denorm = img_left*std + mean
    right_denorm = img_right*std + mean
    
    feature_vec_left = []
    for i in range(len(object_names_left)):
        if graph_modality == "one_hot_labels":
            obj_feature = torch.zeros((len(OBJECT_NAMES_IMAGES)))
            obj_feature[OBJECT_NAMES_IMAGES.index(object_names_left[i])] = 1
            feature_vec_left.append(obj_feature)
        elif graph_modality == "clip_labels":
            clip_embeddings = get_clip_label_embeddings()
            feature_vec_left.append(clip_embeddings[object_names_left[i]])
        elif graph_modality == "random_labels":
            random_embeddings = get_random_label_embeddings()
            feature_vec_left.append(random_embeddings[object_names_left[i]])
        elif graph_modality == "bb_coordinates":
            bbox = objects_left[i]
            pos_feature = torch.tensor(get_bb_pos(bbox)) / 127
            assert pos_feature.shape == (10,)
            feature_vec_left.append(pos_feature)
        elif graph_modality == "cropped_image_feature":
            if isinstance(encoder_model, SimpleImageEncoder):
                if object_names_left[i] in object_names_right:
                    r_obj_idx = object_names_right.index(object_names_left[i])
                    obj_r = torch.tensor(objects_right[r_obj_idx])
                else:
                    obj_r = None
                object_img = crop_and_resize_to_64_for_fusion(left_denorm, right_denorm, torch.tensor(objects_left[i]), obj_r).unsqueeze(0)
            else:
                object_img = crop_and_resize_to_64(left_denorm, torch.tensor(objects_left[i])).unsqueeze(0)
            embedded_bb = encoder_model(object_img).squeeze(0)
            feature_vec_left.append(embedded_bb)
        else:
            print(f"{graph_modality} is not implemented yet")
            assert False
            
    feature_vec_right = []
    for i in range(len(object_names_right)):
        if graph_modality == "one_hot_labels":
            obj_feature = torch.zeros((len(OBJECT_NAMES_IMAGES)))
            obj_feature[OBJECT_NAMES_IMAGES.index(object_names_right[i])] = 1
            feature_vec_right.append(obj_feature)
        elif graph_modality == "clip_labels":
            clip_embeddings = get_clip_label_embeddings()
            feature_vec_right.append(clip_embeddings[object_names_right[i]])
        elif graph_modality == "random_labels":
            random_embeddings = get_random_label_embeddings()
            feature_vec_right.append(random_embeddings[object_names_right[i]])
        elif graph_modality == "bb_coordinates":
            bbox = objects_right[i]
            pos_feature = torch.tensor(get_bb_pos(bbox)) / 127
            assert pos_feature.shape == (10,)
            feature_vec_right.append(pos_feature)
        elif graph_modality == "cropped_image_feature":
            if isinstance(encoder_model, SimpleImageEncoder):
                if object_names_right[i] in object_names_left:
                    l_obj_idx = object_names_left.index(object_names_right[i])
                    obj_l = torch.tensor(objects_left[l_obj_idx])
                else:
                    obj_l = None
                object_img = crop_and_resize_to_64_for_fusion(left_denorm, right_denorm, obj_l, torch.tensor(objects_right[i])).unsqueeze(0)
            else:
                object_img = crop_and_resize_to_64(right_denorm, torch.tensor(objects_right[i])).unsqueeze(0)
            embedded_bb = encoder_model(object_img).squeeze(0)
            feature_vec_right.append(embedded_bb)
        else:
            print(f"{graph_modality} is not implemented yet")
            assert False
    
    return feature_vec_left, feature_vec_right

def calc_bb_mid(coords):
    x_min, y_min, x_max, y_max = coords
    
    x_middle = ((x_max - x_min) / 2) + x_min
    y_middle = ((y_max - y_min) / 2) + y_min
    
    return x_middle, y_middle

def get_bb_pos(coord_list) -> torch.tensor:

    if len(coord_list) == 10:
        pos_feature: torch.Tensor = torch.tensor(coord_list, dtype=torch.float32)
    elif len(coord_list) == 4:        
        x_min, y_min, x_max, y_max = coord_list
        
        x_left_up = x_min
        y_left_up = y_min
        x_right_up = x_max
        y_right_up = y_min
        
        x_left_down = x_min
        y_left_down = y_max
        x_right_down = x_max
        y_right_down = y_max
        
        x_middle, y_middle = calc_bb_mid(coord_list)
    
        pos_feature = torch.tensor([x_left_up, y_left_up, x_right_up, y_right_up, x_left_down, y_left_down, x_right_down, y_right_down, x_middle, y_middle]).type(torch.float32)
    else:
        raise Exception("Bounding box coordinates are not of length 4 or 10!")

    return pos_feature

#################################################################################################################
##################### Methods used for Sparsification Methods ###################################################
#################################################################################################################

RELEVENT_NODES = {
    "PnPCabToCounter":  {'PandaMobile', 'PandaGripper', 'HingeCabinet', 'SingleCabinet', 'Counter', 'obj'},
    "PnPCounterToCab":  {'PandaMobile', 'PandaGripper', 'HingeCabinet', 'SingleCabinet', 'Counter', 'obj'},
    "PnPCounterToMicrowave":  {'PandaMobile', 'PandaGripper', 'Microwave', 'Counter', 'obj'},
    "PnPCounterToSink":  {'PandaMobile', 'PandaGripper', 'Sink', 'Counter', 'obj'},
    "PnPCounterToStove":  {'PandaMobile', 'PandaGripper', 'Stove', 'Counter', 'obj'},
    "PnPMicrowaveToCounter":  {'PandaMobile', 'PandaGripper', 'Microwave', 'Counter', 'obj'},
    "PnPSinkToCounter":  {'PandaMobile', 'PandaGripper', 'Sink', 'Counter', 'obj'},
    "PnPStoveToCounter":  {'PandaMobile', 'PandaGripper', 'Stove', 'Counter', 'obj'},
    "OpenSingleDoor":  {'PandaMobile', 'PandaGripper'},
    "OpenDoubleDoor":  {'PandaMobile', 'PandaGripper'},
    "CloseDoubleDoor":  {'PandaMobile', 'PandaGripper'},
    "CloseSingleDoor":  {'PandaMobile', 'PandaGripper'},
    "OpenDrawer":      {'PandaMobile', 'PandaGripper', 'Drawer', 'Counter'},
    "CloseDrawer":      {'PandaMobile', 'PandaGripper', 'Drawer', 'Counter'},
    "TurnOnStove":    {'PandaMobile', 'PandaGripper', 'Stove'},
    "TurnOffStove":    {'PandaMobile', 'PandaGripper', 'Stove'},
    "TurnOnSinkFaucet":    {'PandaMobile', 'PandaGripper', 'Sink'},
    "TurnOffSinkFaucet":    {'PandaMobile', 'PandaGripper', 'Sink'},
    "TurnSinkSpout":    {'PandaMobile', 'PandaGripper', 'Sink'},
    "CoffeePressButton":    {'PandaMobile', 'PandaGripper', 'CoffeeMachine'},
    "TurnOnMicrowave":    {'PandaMobile', 'PandaGripper', 'Microwave'},
    "TurnOffMicrowave":    {'PandaMobile', 'PandaGripper', 'Microwave'},
    "CoffeeServeMug":    {'PandaMobile', 'PandaGripper', 'Counter', 'CoffeeMachine', 'obj'},
    "CoffeeSetupMug":    {'PandaMobile', 'PandaGripper', 'Counter', 'CoffeeMachine', 'obj'},
}

def is_node_relevant_for_task(node_name, task_name,     lang_goal):
    is_relevant = False
    if node_name in RELEVENT_NODES[task_name]:
        is_relevant = True
    else:
        if task_name in ["CloseSingleDoor", "OpenSingleDoor", "CloseDoubleDoor", "OpenDoubleDoor"]:
            # 'door_obj' is the fixed, task-agnostic node name for the actual interactive door
            # panel across every door task instance (OBJECT_NAMES_IMAGES, position 25) -
            # regardless of which fixture (microwave, cabinet, ...) it belongs to.
            if node_name == "door_obj":
                is_relevant = True
            else:
                # Fallback: also count the specific fixture body named in the instruction
                # (e.g. 'Microwave' for "close the microwave door") as relevant.
                words = (lang_goal or "").split(" ")
                if len(words) >= 2:
                    obj_door_to_close = words[-2].lower()
                    if obj_door_to_close in node_name.lower():
                        is_relevant = True

    return is_relevant

def get_num_relevant_nodes_per_task(task_name):
    if task_name in ["CloseSingleDoor", "OpenSingleDoor", "CloseDoubleDoor", "OpenDoubleDoor"]:
        return len(RELEVENT_NODES[task_name]) + 1
    else:
        return len(RELEVENT_NODES[task_name])
    
    