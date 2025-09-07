import runpod
import torch
import os
import cv2
import numpy as np
from model import Your3DMMModel  # adjust based on repo
from utils import save_obj        # adjust based on repo

import sys 
sys.path.append(os.path.join(os.getcwd(), "third_party/3DMM-Fitting-Pytorch"))

from utils import util
from model import BaselFaceModel   # example, adjust as needed
import cv2

# Initialize 3DMM
bfm = BaselFaceModel(MODEL_PATH, EXP_PATH)
# Load 3DMM assets from RunPod volume
MODEL_PATH = "/runpod-volume/01_MorphableModel.mat"
EXP_PATH = "/runpod-volume/Exp_Pca.bin"

# Initialize 3DMM
bfm = BaselFaceModel(MODEL_PATH, EXP_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Your3DMMModel(MODEL_PATH, EXP_PATH).to(device)
model.eval()

def handler(event):
    """
    event = {
        "input": {
            "image": "base64_string OR url"
        }
    }
    """
    try:
        image_path = event["input"]["image"]

        # Load input face image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not load image"}

        # Run fitting
        verts, faces = model.fit(img)

        # Save as .obj for Unity
        output_path = "/runpod-volume/output.obj"
        save_obj(output_path, verts, faces)

        return {"output_obj": output_path}

    except Exception as e:
        return {"error": str(e)}

# Start RunPod
runpod.serverless.start({"handler": handler})
