import os
import io
import base64
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
import face_alignment

from core import get_recon_model
import core.utils as utils
import core.losses as losses

# Convenience: Torch device selection
_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Where BFM assets live (mounted Network Volume)
BFM_DIR = os.environ.get('BFM_DIR', '/runpod-volume/BFM')
REPO_ROOT = os.environ.get('REPO_ROOT', '/workspace/third_party/3DMM-Fitting-Pytorch')

# Ensure converted BFM file exists (generate if raw assets are present)
def ensure_bfm_converted():
    converted = os.path.join(BFM_DIR, 'BFM09_model_info.mat')
    if os.path.exists(converted):
        return converted
    raw_mat = os.path.join(BFM_DIR, '01_MorphableModel.mat')
    raw_exp = os.path.join(BFM_DIR, 'Exp_Pca.bin')
    if os.path.exists(raw_mat) and os.path.exists(raw_exp):
        # Run the repo's converter
        import subprocess
        print('[BFM] Converting BFM09 data...')
        subprocess.check_call([
            'python3', os.path.join(REPO_ROOT, 'convert_bfm09_data.py')
        ], cwd=REPO_ROOT)
        assert os.path.exists(converted), 'BFM conversion did not produce BFM09_model_info.mat'
        return converted
    raise FileNotFoundError(
        f"Missing BFM assets. Place '01_MorphableModel.mat' and 'Exp_Pca.bin' under {BFM_DIR}")

# Core inference adapted from fit_single_img.py
def run_inference(
    image_bgr: np.ndarray,
    tar_size: int = 224,
    rf_lr: float = 1e-2,
    nrf_lr: float = 1e-2,
    first_rf_iters: int = 100,
    first_nrf_iters: int = 200,
    padding_ratio: float = 0.35,
    recon_model_name: str = 'BFM09',
):
    # 1) Pre-flight: BFM
    ensure_bfm_converted()

    # 2) Detectors
    mtcnn = MTCNN(device=_DEVICE, select_largest=False)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    # 3) Recon model
    recon_model = get_recon_model(model=recon_model_name, device=_DEVICE, batch_size=1, img_size=tar_size)

    # 4) Face detect & crop
    img_rgb = image_bgr[:, :, ::-1]
    h, w = img_rgb.shape[:2]
    bboxes, probs = mtcnn.detect(img_rgb)
    if bboxes is None or len(bboxes) == 0:
        return {"status": "no_face"}
    bbox = utils.pad_bbox(bboxes[0], (w, h), padding_ratio)
    face_w, face_h = int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
    assert face_w == face_h, 'Expected square crop from pad_bbox.'

    face_img = img_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
    resized_face_img = cv2.resize(face_img, (tar_size, tar_size))

    lms = fa.get_landmarks_from_image(resized_face_img)[0]
    lms = torch.tensor(lms[:, :2][None, ...], dtype=torch.float32, device=_DEVICE)
    img_tensor = torch.tensor(resized_face_img[None, ...], dtype=torch.float32, device=_DEVICE)

    lm_weights = utils.get_lm_weights(_DEVICE)

    # 5) Rigid fitting
    rigid_opt = torch.optim.Adam([recon_model.get_rot_tensor(), recon_model.get_trans_tensor()], lr=rf_lr)
    for _ in range(first_rf_iters):
        rigid_opt.zero_grad()
        pred = recon_model(recon_model.get_packed_tensors(), render=False)
        lm_loss_val = losses.lm_loss(pred['lms_proj'], lms, lm_weights, img_size=tar_size)
        (lm_loss_val).backward()
        rigid_opt.step()

    # 6) Non-rigid fitting
    nrf_opt = torch.optim.Adam([
        recon_model.get_id_tensor(),
        recon_model.get_exp_tensor(),
        recon_model.get_gamma_tensor(),
        recon_model.get_tex_tensor(),
        recon_model.get_rot_tensor(),
        recon_model.get_trans_tensor()], lr=nrf_lr)

    for _ in range(first_nrf_iters):
        nrf_opt.zero_grad()
        pred = recon_model(recon_model.get_packed_tensors(), render=True)
        rendered = pred['rendered_img']
        mask = (rendered[:, :, :, 3].detach() > 0)
        photo_loss_val = losses.photo_loss(rendered[:, :, :, :3], img_tensor, mask)
        lm_loss_val = losses.lm_loss(pred['lms_proj'], lms, lm_weights, img_size=tar_size)
        id_reg = losses.get_l2(recon_model.get_id_tensor())
        exp_reg = losses.get_l2(recon_model.get_exp_tensor())
        tex_reg = losses.get_l2(recon_model.get_tex_tensor())
        tex_loss = losses.reflectance_loss(pred['face_texture'], recon_model.get_skinmask())
        total = lm_loss_val*1.0 + id_reg*1e-4 + exp_reg*1e-4 + tex_reg*1e-4 + tex_loss*1.0 + photo_loss_val*1.0
        total.backward()
        nrf_opt.step()

    # 7) Export (coeffs + mesh)
    with torch.no_grad():
        coeffs_tensor = recon_model.get_packed_tensors()
        pred = recon_model(coeffs_tensor, render=True)
        vs = pred['vs'].cpu().numpy().squeeze()
        tri = (pred['tri'].cpu().numpy().squeeze()) + 1  # OBJ is 1-indexed
        color = pred['color'].cpu().numpy().squeeze()

        # Save artifacts to temp files
        os.makedirs('/tmp/out', exist_ok=True)
        base = 'result'
        coeff_path = f'/tmp/out/{base}_coeffs.npy'
        obj_path = f'/tmp/out/{base}_mesh.obj'
        np.save(coeff_path, coeffs_tensor.detach().cpu().numpy().squeeze())
        utils.save_obj(obj_path, vs, tri, color)

        # Read back and base64-encode
        with open(coeff_path, 'rb') as f:
            coeff_b64 = base64.b64encode(f.read()).decode('ascii')
        with open(obj_path, 'rb') as f:
            obj_b64 = base64.b64encode(f.read()).decode('ascii')

    return {
        "status": "ok",
        "coeffs_npy_b64": coeff_b64,
        "mesh_obj_b64": obj_b64,
    }

# Helper to accept image via URL-safe base64 or raw bytes

def decode_image(image_b64: str = None, image_bytes: bytes = None):
    if image_bytes is None and image_b64 is None:
        raise ValueError('Provide image_b64 or image_bytes')
    data = base64.b64decode(image_b64) if image_b64 else image_bytes
    image = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('Could not decode image')
    return bgr