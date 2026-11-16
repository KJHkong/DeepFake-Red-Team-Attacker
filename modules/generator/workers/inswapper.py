"""Official INSwapper + existing CPU analysis loader; no downloads.

Source: https://github.com/deepinsight/insightface/tree/master/examples/in_swapper
Loader matches the preserved reproduction/run_inference.py (buffalo_l/r50).
"""
import importlib
from pathlib import Path
import sys
import types


def generate(job, failure):
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    repo, cache = Path(job["repo"]), Path(job["checkpoint"])
    base = repo.parents[2]
    sys.path.insert(0, str(base/"Synthetic/Avatar_Digital_Human/LivePortrait"))
    from src.utils.dependencies.insightface.app import FaceAnalysis
    from src.utils.dependencies.insightface.model_zoo import model_zoo
    cv2.setNumThreads(4)
    original = model_zoo.PickableInferenceSession.__init__
    def session_init(self, model_path, **kwargs):
        opt = ort.SessionOptions(); opt.intra_op_num_threads = 4; opt.inter_op_num_threads = 1
        kwargs["sess_options"] = opt
        original(self, model_path, **kwargs)
    model_zoo.PickableInferenceSession.__init__ = session_init
    # These files must already exist; prevents FaceAnalysis's download fallback.
    for name in ("det_10g.onnx", "w600k_r50.onnx"):
        if not (cache/"face/models/buffalo_l"/name).is_file():
            raise failure("generation_failed")
    app = FaceAnalysis(name="buffalo_l", root=str(cache/"face"), allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(512,512), det_thresh=.5)
    for name, rel in [("_official_insightface", ""), ("_official_insightface.utils", "utils"), ("_official_insightface.model_zoo", "model_zoo")]:
        module = types.ModuleType(name)
        module.__path__ = [str(repo/"python-package/insightface"/rel)]
        sys.modules[name] = module
    official = importlib.import_module("_official_insightface.model_zoo.inswapper")
    opt = ort.SessionOptions(); opt.intra_op_num_threads = 4; opt.inter_op_num_threads = 1
    session = ort.InferenceSession(str(cache/"inswapper_128.onnx"), sess_options=opt, providers=["CPUExecutionProvider"])
    swapper = official.INSwapper(str(cache/"inswapper_128.onnx"), session=session)
    images, faces, counts = {}, {}, {}
    for role in ("source", "target"):
        image = cv2.imread(job["inputs"][role])
        if image is None:
            raise failure("invalid_input")
        found = app.get(image)
        if not found:
            raise failure("no_face")
        images[role] = image
        faces[role] = max(found, key=lambda f: float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
        counts[role] = len(found)
    merged = swapper.get(images["target"], faces["target"], faces["source"], paste_back=True)
    if not np.isfinite(merged).all():
        raise failure("generation_failed")
    Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)).save(job["output"])
    return {"onnxruntime": ort.__version__, "providers": session.get_providers(), "face_counts": counts,
        "face_selection": "largest", "native_face_size": [128,128], "paste_back": True,
        "seed_used": False, "recognition_model": "buffalo_l/w600k_r50.onnx"}
