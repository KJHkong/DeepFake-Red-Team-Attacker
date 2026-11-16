"""Original MobileFaceSwap forward/align/dealign in a Paddle2.4.2 compat runtime.

Sources: https://github.com/Seanseattle/MobileFaceSwap/blob/master/image_test.py
https://github.com/Seanseattle/MobileFaceSwap/blob/master/utils/align_face.py
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from time import perf_counter
import traceback


def configure_cpu_sessions(ort):
    original = ort.InferenceSession.__init__
    def session_init(self, path, sess_options=None, providers=None, provider_options=None, **kwargs):
        options = sess_options or ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        return original(self, path, sess_options=options, providers=['CPUExecutionProvider'], **kwargs)
    ort.InferenceSession.__init__ = session_init


def run_native(job, failure):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import paddle
    from PIL import Image

    repo, cache = Path(job['repo']), Path(job['checkpoint'])
    folder = Path(job['output']).parent
    cv2.setNumThreads(4)
    # Original ONNX detection/model loader, with bounded CPU thread pools.
    configure_cpu_sessions(ort)
    sys.path.insert(0, str(repo))
    from image_test import get_id_emb
    from models.model import FaceSwap
    from models.arcface import IRBlock, ResNet
    from utils.prepare_data import LandmarkModel
    from utils.align_face import align_img, back_matrix, dealign
    from utils.util import cv2paddle, paddle2cv

    paddle.set_device('gpu:0')
    paddle.seed(42)
    detector = LandmarkModel(name='landmarks', root=str(cache))
    detector.prepare(ctx_id=-1, det_thresh=.6, det_size=(640,640))
    images, points, aligned, inverse = {}, {}, {}, {}
    preprocess = perf_counter()
    for role in ('source', 'target'):
        image = cv2.imread(job['inputs'][role])
        if image is None:
            raise failure('invalid_input')
        keypoints = detector.get(image)
        if keypoints is None:
            raise failure('no_face')
        if keypoints.shape != (5,2) or not np.isfinite(keypoints).all():
            raise failure('generation_failed')
        images[role], points[role] = image, keypoints
        aligned[role], inverse[role] = align_img(image, keypoints, 224)
        cv2.imwrite(str(folder/f'mobilefaceswap_{role}_aligned_1109.png'), aligned[role])
    cv2.imwrite(str(folder/'mobilefaceswap_source_identity_1109.png'), cv2.resize(aligned['source'], (112,112)))
    preprocess_ms = (perf_counter()-preprocess)*1000
    transform = dict(source_keypoints=points['source'].tolist(), target_keypoints=points['target'].tolist(),
        source_inverse=inverse['source'].tolist(), target_inverse=inverse['target'].tolist(),
        target_forward=back_matrix(inverse['target']).tolist(),
        coordinate_system='original decoded Target pixels; official least-squares affine and pseudoinverse',
        source_aligned_size=[224,224], identity_input_size=[112,112], target_aligned_size=[224,224])
    transform_path = folder/'mobilefaceswap_transform_1109.json'
    transform_path.write_text(json.dumps(transform, indent=2)+'\n')
    loading = perf_counter()
    faceswap = FaceSwap(True)
    identity = ResNet(block=IRBlock, layers=[3,4,23,3])
    identity.set_dict(paddle.load(str(cache/'arcface.pdparams')))
    identity.eval()
    weight = paddle.load(str(cache/'MobileFaceSwap_224.pdparams'))
    model_load_ms = (perf_counter()-loading)*1000
    forward = perf_counter()
    # Evaluation-only gradient disabling leaves official values and kernels intact.
    with paddle.no_grad():
        embedding, feature = get_id_emb(identity, str(folder/'mobilefaceswap_source_aligned_1109.png'))
        faceswap.set_model_param(embedding, feature, model_weight=weight)
        faceswap.eval()
        output, mask = faceswap(cv2paddle(aligned['target']))
        result = paddle2cv(output)
        raw_mask = np.transpose(mask[0].numpy(), (1,2,0))
    forward_ms = (perf_counter()-forward)*1000
    if not np.isfinite(result).all() or not np.isfinite(raw_mask).all():
        raise failure('generation_failed')
    # Preserve exact float intermediates; display PNGs are separate projections.
    np.save(folder/'mobilefaceswap_aligned_float_1109.npy', result)
    np.save(folder/'mobilefaceswap_mask_float_1109.npy', raw_mask)
    cv2.imwrite(str(folder/'mobilefaceswap_aligned_output_1109.png'), result)
    cv2.imwrite(str(folder/'mobilefaceswap_learned_mask_1109.png'), np.clip(raw_mask[:,:,0]*255,0,255).astype('uint8'))
    post = perf_counter()
    # The original function mutates its mask; give it a copy of the same values.
    final = dealign(result, images['target'], inverse['target'], raw_mask.copy())
    if final.shape != images['target'].shape or not np.isfinite(final).all():
        raise failure('generation_failed')
    if not cv2.imwrite(job['output'], final):
        raise failure('generation_failed')
    post_ms = (perf_counter()-post)*1000
    artifacts = []
    for name in ['source_aligned','source_identity','target_aligned','aligned_output','learned_mask']:
        path = folder/f'mobilefaceswap_{name}_1109.png'
        with Image.open(path) as image:
            width, height = image.size
            image.load()
        data = path.read_bytes()
        artifacts.append(dict(path=str(path), media_type='image/png', width=width, height=height,
            size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return dict(paddle=paddle.__version__, onnxruntime=ort.__version__, numpy=np.__version__, opencv=cv2.__version__,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], framework_device=paddle.get_device(),
        detector_provider='CPUExecutionProvider', face_selection='official highest detection confidence',
        detector_size=[640,640], detector_threshold=.6, native_face_size=[224,224], identity_input_size=[112,112],
        paste_back=True, original_target_size=[final.shape[1],final.shape[0]], modifiers=False,
        seed=42, seed_scope='fixed framework initialization only; no latent sampling or score-based selection',
        precision='official float32', original_model_math_unchanged=True, output_format='lossless PNG before any JPEG encoding',
        preprocess_ms=preprocess_ms, model_load_ms=model_load_ms, identity_setup_forward_ms=forward_ms, paste_save_ms=post_ms,
        intermediates=artifacts, transform_path=str(transform_path), transform_sha256=hashlib.sha256(transform_path.read_bytes()).hexdigest(),
        exact_float_evidence={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('mobilefaceswap_*_float_1109.npy')})


def main():
    # Conda's private CUDA/cuDNN libraries apply to this Worker process only.
    lib = str(Path(sys.executable).parent.parent/'lib')
    if os.environ.get('MOBILEFACE_PRIVATE_LIBS_READY') != '1':
        env = dict(os.environ, MOBILEFACE_PRIVATE_LIBS_READY='1',
            LD_LIBRARY_PATH=lib+':'+os.environ.get('LD_LIBRARY_PATH',''), FLAGS_allocator_strategy='auto_growth')
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]], env)
    from run import check_gpu, WorkerFailure
    job = json.loads(Path(sys.argv[1]).read_text())
    start = perf_counter()
    result = dict(status='failed', error_code=None, metadata=dict(python=platform.python_version(), python_executable=sys.executable))
    stop = threading.Event()
    memory = []
    def observe():
        while not stop.is_set():
            try:
                raw = subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
                    '--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True, timeout=3)
                memory.append(int(raw.strip()))
            except (ValueError, OSError, subprocess.SubprocessError):
                pass
            stop.wait(.5)
    observer = None
    try:
        if job['tool'] != 'mobilefaceswap' or job['parameters']:
            raise WorkerFailure('invalid_input')
        result['metadata'].update(check_gpu())
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        result['metadata'].update(run_native(job, WorkerFailure))
        result['status'] = 'success'
    except WorkerFailure as e:
        result['error_code'] = str(e)
    except Exception:
        result['error_code'] = 'generation_failed'
        traceback.print_exc()
    finally:
        stop.set()
        if observer:
            observer.join(timeout=4)
    result['metadata'].update(elapsed_ms=(perf_counter()-start)*1000,
        sampled_device_memory_peak_mib=max(memory) if memory else None,
        memory_measurement='nvidia-smi device memory sampled every0.5s while exclusively admitted; not an exact allocation peak')
    Path(job['reply']).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
