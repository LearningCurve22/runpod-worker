import os
import runpod
from app.api_infer import run_inference, decode_image

# Optional: allow overriding defaults via environment
DEFAULTS = {
    'tar_size': int(os.environ.get('TAR_SIZE', 224)),
    'rf_lr': float(os.environ.get('RF_LR', 1e-2)),
    'nrf_lr': float(os.environ.get('NRF_LR', 1e-2)),
    'first_rf_iters': int(os.environ.get('FIRST_RF_ITERS', 100)),
    'first_nrf_iters': int(os.environ.get('FIRST_NRF_ITERS', 200)),
    'padding_ratio': float(os.environ.get('PADDING_RATIO', 0.35)),
    'recon_model_name': os.environ.get('RECON_MODEL', 'BFM09'),
}

# Main handler for RunPod Serverless

def handler(event):
    inp = event.get('input', {})

    # Accept either base64 image or direct bytes (if using binary bridge)
    image_b64 = inp.get('image_b64')
    image_bytes = None  # not used by default

    # Merge overrides
    args = DEFAULTS.copy()
    args.update({k: v for k, v in inp.items() if k in DEFAULTS})

    try:
        bgr = decode_image(image_b64=image_b64, image_bytes=image_bytes)
        result = run_inference(
            bgr,
            tar_size=args['tar_size'],
            rf_lr=args['rf_lr'],
            nrf_lr=args['nrf_lr'],
            first_rf_iters=args['first_rf_iters'],
            first_nrf_iters=args['first_nrf_iters'],
            padding_ratio=args['padding_ratio'],
            recon_model_name=args['recon_model_name'],
        )
        return {"output": result}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})