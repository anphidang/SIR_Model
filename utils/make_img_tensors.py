"""Pack the per-demo camera images into img_tensor_demo_<N>.pth, the file layout
RoboCasaDataset.get_image_obs expects (and that no script in this repo produced - the files were
deleted after the graph datasets were uploaded, see the note on create_graphs_and_save).

Layout: (steps, 3*3*128*128), cameras stacked as [inhand, left, right] (get_image_obs indexes
[:,0]=inhand, [:,1]=left, [:,2]=right). Pixels go through exactly the transform that
RoboCasaKitchenTester applies to rollout observations (ToTensor -> Resize -> Normalize), so
training and rollout see the same input distribution - including its std=[.., .., 0.255].
Stored as float16 (halves the ~8 GB float32 footprint; get_image_obs casts back to float).

Usage: python utils/make_img_tensors.py --task CloseSingleDoor
"""
import argparse
import os

import h5py
import torch
from torchvision import transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_path", default="/home/an-phi/Dokumente/rob/SIR_Model/data/")
    args = ap.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
    ])
    task_dir = os.path.join(args.data_path, args.task)
    f = h5py.File(os.path.join(task_dir, "demo_gentex_im128_randcams.hdf5"), "r")
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    # same numbering the dataset uses: demos may start at demo_1 (human demos) -> file index 0
    offset = int(keys[0].split("_")[-1])
    for key in keys:
        obs = f["data"][key]["obs"]
        cams = [obs["robot0_eye_in_hand_image"][()], obs["robot0_agentview_left_image"][()],
                obs["robot0_agentview_right_image"][()]]
        steps = cams[0].shape[0]
        per_step = []
        for t in range(steps):
            per_step.append(torch.cat([transform(c[t]).reshape(-1) for c in cams]))
        out = torch.stack(per_step).to(torch.float16)
        idx = int(key.split("_")[-1]) - offset
        torch.save(out, os.path.join(task_dir, f"img_tensor_demo_{idx}.pth"))
        print(key, "->", idx, tuple(out.shape))


if __name__ == "__main__":
    main()
