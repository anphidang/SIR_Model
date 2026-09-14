import torch

from utils.generate_3d_bb_dataset_robocasa import BB3D_FEATURE_DIM
from utils.generate_graph_dataset_robocasa import CLIP_LABEL_DIM

GRAPH_MODALITY_LIST = ["one_hot_labels", "clip_labels", "bb_coordinates", "cropped_image_feature", "bb3d_coordinates"]

def combine_graph_modalities(graph_data, graph_mod, idx=None, j=None):
    if idx is None and j is None:
        graphs = [graph_data[mod] for mod in graph_mod]
    else:
        graphs = [graph_data[mod][idx][j] for mod in graph_mod]

    # Different modalities can see different subsets of objects per frame (e.g. bb3d_coordinates
    # fuses both static cams while cropped_image_feature only sees objects segmented in a single
    # view), so - like fuse_graphs() does for left/right of the same modality - align every
    # modality's nodes onto the union of node names (zero-padding missing ones) before
    # concatenating features, instead of assuming identical node order/count across modalities.
    all_names = sorted(set().union(*(g.node_names for g in graphs)))
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    num_nodes = len(all_names)

    feat_blocks = []
    for g in graphs:
        block = torch.zeros((num_nodes, g.x.shape[1]), device=g.x.device, dtype=g.x.dtype)
        for src_i, name in enumerate(g.node_names):
            block[name_to_idx[name]] = g.x[src_i]
        feat_blocks.append(block)

    fused_x = torch.cat(feat_blocks, dim=-1)

    src, dst = torch.meshgrid(
        torch.arange(num_nodes, device=fused_x.device),
        torch.arange(num_nodes, device=fused_x.device),
        indexing="ij"
    )
    src = src.flatten()
    dst = dst.flatten()
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    edge_index = torch.stack((src, dst), dim=0)
    edge_attr = torch.ones(edge_index.shape[1], device=fused_x.device, dtype=torch.float32)

    base_graph = type(graphs[0])(
        x=fused_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_names=all_names,
    )

    return base_graph

def fuse_graphs(graph_data_left, graph_data_right, mod, step_idx=None, is_cropped_fusion=False):
    if step_idx is None:
        left_graph = graph_data_left[mod]
        right_graph = graph_data_right[mod]
    else:
        left_graph = graph_data_left[mod][step_idx]
        right_graph = graph_data_right[mod][step_idx]
    
    left_names = left_graph.node_names
    right_names = right_graph.node_names
    all_names = sorted(list(set(left_names) | set(right_names)))
    
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    num_nodes = len(all_names)
    
    feat_dim_l = left_graph.x.shape[1]
    feat_dim_r = right_graph.x.shape[1]
    
    # Create zero-filled tensors for the fused graph
    # Shape: [Total_Unique_Nodes, Left_Dim]
    x_l_mapped = torch.zeros((num_nodes, feat_dim_l), device=left_graph.x.device, dtype=left_graph.x.dtype)
    # Shape: [Total_Unique_Nodes, Right_Dim]
    x_r_mapped = torch.zeros((num_nodes, feat_dim_r), device=right_graph.x.device, dtype=right_graph.x.dtype)
    
    # Map Left Features
    for src_i, name in enumerate(left_names):
        target_i = name_to_idx[name]
        x_l_mapped[target_i] = left_graph.x[src_i]
        
    # Map Right Features
    for src_i, name in enumerate(right_names):
        target_i = name_to_idx[name]
        x_r_mapped[target_i] = right_graph.x[src_i]

    fused_x, bb_index = handle_fusing(x_l_mapped, x_r_mapped, mod, is_cropped_fusion)

    src, dst = torch.meshgrid(
        torch.arange(num_nodes, device=left_graph.x.device),
        torch.arange(num_nodes, device=left_graph.x.device),
        indexing="ij"
    )
    
    src = src.flatten()
    dst = dst.flatten()
    
    # Filter out self-loops (i != j)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    fused_edge_index = torch.stack((src, dst), dim=0)
    
    # if "bb_coordinates" in mod:
    #     fused_edge_attr = calculate_weight_dim_distance(fused_x, bb_index, src, dst)
    # else:
    fused_edge_attr = torch.ones(fused_edge_index.shape[1], device=left_graph.x.device, dtype=torch.float32)

    fused_graph = type(left_graph)(
        x=fused_x,
        edge_index=fused_edge_index,
        edge_attr=fused_edge_attr,
        node_names=all_names  # Store the new unified list of names
    )
    
    return fused_graph

def handle_fusing(left, right, mod, cropped_fusion):
    # Look up each modality by name, not by GRAPH_MODALITY_LIST position - a positional
    # mapping silently breaks (wrong tag, wrong length, or a missing branch entirely) whenever
    # a modality is inserted anywhere but the end of that list. This broke exactly that way
    # when "clip_labels" was inserted at index 1: every following branch's slice tag/length
    # was off by one and "bb3d_coordinates" (originally index 3) dropped out of the checks.
    #
    # Rank = the substring's start index in `mod` (e.g. "bb_coordinates_cropped_image_feature_
    # one_hot_labels"), so modalities are sliced out in the same order they were concatenated
    # in - this must work for any number of joined modalities, not just two. The previous
    # `mod.split(name).index('')` trick only ever found a rank when `name` was a prefix (split
    # -> ['', rest]) or suffix (split -> [rest, '']) of `mod`; for 3+ joined modalities the
    # middle one splits into two non-empty parts and `.index('')` raises `ValueError`.
    active_mods = []

    if "one_hot_labels" in mod:
        ohl_index = mod.find("one_hot_labels")
        ohl_length = 37
        active_mods.append((ohl_index, 'ohl', ohl_length))
    if "clip_labels" in mod:
        clip_index = mod.find("clip_labels")
        active_mods.append((clip_index, 'clip', CLIP_LABEL_DIM))
    if "bb_coordinates" in mod:
        bb_index = mod.find("bb_coordinates")
        bb_length = 10
        active_mods.append((bb_index, 'bb', bb_length))
    if "cropped_image_feature" in mod:
        cropped_index = mod.find("cropped_image_feature")
        # We use -1 as placeholder for dynamic length
        active_mods.append((cropped_index, 'crop', -1))
    if "bb3d_coordinates" in mod:
        bb3d_index = mod.find("bb3d_coordinates")
        active_mods.append((bb3d_index, 'bb3d', BB3D_FEATURE_DIM))

    # 2. Sort by rank (position in the tensor)
    active_mods.sort(key=lambda x: x[0])
    
    # 3. Calculate the dynamic length of 'cropped_image_feature'
    total_input_len = left.shape[-1]
    known_len = sum(m[2] for m in active_mods if m[2] != -1)
    crop_len = total_input_len - known_len
    
    # 4. Iterate, slice, and process
    fused_parts = []
    current_ptr = 0
    
    bb_index = -1
    
    for rank, name, length in active_mods:
        # Resolve actual length if dynamic
        eff_len = crop_len if length == -1 else length
        
        # Slice the current modality from both views
        l_slice = left[..., current_ptr : current_ptr + eff_len]
        r_slice = right[..., current_ptr : current_ptr + eff_len]
        
        if name == 'bb':
            # BB: User wants BOTH values included -> Concatenate (Length becomes 2x)
            fused_parts.append(torch.cat((l_slice, r_slice), dim=-1))
            bb_index = current_ptr
        elif name == 'crop':
            if cropped_fusion:
                fused_parts.append(l_slice)
            else:
                fused_parts.append(torch.cat((l_slice, r_slice), dim=-1))
        elif name == 'bb3d':
            # World-frame box, already fused across both static cams -> take one copy,
            # same as OHL (concatenating would just duplicate identical values).
            fused_parts.append(l_slice)
        else:
            # OHL: ONE value (identical) -> Take Left
            fused_parts.append(l_slice)
            
        current_ptr += eff_len
    
    # 5. Reassemble the fused features
    return torch.cat(fused_parts, dim=-1), bb_index


def pad_sequence(sequence, target_length):
    """Helper to pad a sequence by repeating the last frame if it's too short."""
    current_len = sequence.shape[0]
    if current_len < target_length:
        pad_qty = target_length - current_len
        last_frame = sequence[-1].unsqueeze(0)
        padding = last_frame.repeat(pad_qty, *([1]*(len(sequence.shape)-1)))
        return torch.cat((sequence, padding), dim=0)
    return sequence

def convert_weight_to_dim_distance(graph_data, graph_mod):
    for i in range(len(graph_data['bb_coordinates'])):
        for j in range(len(graph_data['bb_coordinates'][i])):
            middle_points = graph_data['bb_coordinates'][i][j].x[:,8:]
            
            row, col = graph_data['bb_coordinates'][i][j].edge_index
            
            source_points = middle_points[row] 
            target_points = middle_points[col]
            
            distance = torch.abs(source_points - target_points)
            
            for mod in graph_mod:
                graph_data[mod][i][j].edge_attr = distance
    return graph_data

def calculate_weight_dim_distance(graph_data, start_index, row, col):
    # TODO fix for all possible combinations
    middle_points_left = graph_data[:,start_index+8:start_index+10]
    middle_points_right = graph_data[:,start_index+18:start_index+20]
    
    source_points_left = middle_points_left[row]
    target_points_left = middle_points_left[col]
    
    source_points_right = middle_points_right[row]
    target_points_right = middle_points_right[col]
    
    distance_left = torch.abs(source_points_left - target_points_left)
    distance_right = torch.abs(source_points_right - target_points_right)
    
    distance = torch.concat((distance_left, distance_right), dim=-1)
    
    return distance