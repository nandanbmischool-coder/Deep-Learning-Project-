# Architecture notes

**Process model.** The API (FastAPI/uvicorn) never trains in-process. `TrainingManager` spawns `python -m app.ml.train` as a subprocess writing `status.json`, `progress.jsonl` and `train.log` into a fresh run folder (`data/artifacts/runs/<arch>-<timestamp>`). The UI polls `GET /training/jobs/{id}`, which reads those files, so progress is real. A watcher thread registers the finished model (+ evaluation) in the DB. Only one job may run; a new run folder per job prevents checkpoint collisions; writes are atomic (tmp + rename).

**Model registry.** `model_versions` rows point at run folders (relative paths). The first registered model is activated automatically; admins can switch. `ModelService` lazily loads the active checkpoint (`weights_only=True`) and serialises inference because Grad-CAM hooks are per-model state.

**Grad-CAM layer choice.** torchvision DenseNet applies `F.relu(inplace=True)` after `features.norm5`, which interferes with tensor hooks, so the hook sits on `features.denseblock4` (last convolutional feature map; norm5 and ReLU are downstream, so gradients still flow through them). ResNet50 uses `layer4`, MobileNetV3-Large `features.16`. `tests/test_images_and_predictions.py` instantiates every architecture and checks the layer exists and yields a valid map.

**Evaluation protocol.** Validation (carved from `train`, stratified by label, grouped by parsed patient id) selects the checkpoint (min val loss) and the decision threshold (Youden's J). The official `test` split is scored once, with a fixed threshold. Bootstrap CIs resample test images; ROC/PR use continuous scores.

**Data stores.** SQL for users/analyses/models/evaluations/jobs/audit; images on disk under `data/media/analyses/<user>/<analysis>/` with only relative paths in the DB; assets are served through authenticated, owner-checked endpoints.

**Auth.** HttpOnly cookie JWT + readable CSRF cookie echoed in `X-CSRF-Token`. The Vite dev server proxies `/api` so everything is same-origin; in Docker nginx does the same.
