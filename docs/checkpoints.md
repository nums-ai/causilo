# Checkpoints

`model_version` selects the hosted checkpoint for the estimator's task.
Both weights and configuration are pinned to the revision in this table.

| Model | Task | Hugging Face revision | Files | Access |
| --- | --- | --- | --- | --- |
| 1.5 (default) | Classification | [4ca2275](https://huggingface.co/nums-ai/causilo-1.5/tree/4ca2275bae85014a19422e8c18fcf8f24c5c14e1/classifier) | `classifier/` | Public |
| 1.5 (default) | Regression | [4ca2275](https://huggingface.co/nums-ai/causilo-1.5/tree/4ca2275bae85014a19422e8c18fcf8f24c5c14e1/regressor) | `regressor/` | Public |
| 1.0 | Classification | [94f2bd9](https://huggingface.co/nums-ai/causilo/tree/94f2bd91db0737d4da59f347910662905ecb5a09/classifier) | `classifier/` | Public |
| 1.0 | Regression | [94f2bd9](https://huggingface.co/nums-ai/causilo/tree/94f2bd91db0737d4da59f347910662905ecb5a09/regressor) | `regressor/` | Public |

The default and Light modes use the same checkpoint for each task.

## Model licenses

- **Causilo 1.0** classifier and regressor weights are covered by the
  [license included in the original Causilo 1.0 release](https://huggingface.co/nums-ai/causilo/resolve/94f2bd91db0737d4da59f347910662905ecb5a09/LICENSE).
- **Causilo 1.5 Classifier** and **Causilo 1.5 Regressor** weights are covered by
  the [Nums AI Model License](https://huggingface.co/nums-ai/causilo-1.5/blob/4ca2275bae85014a19422e8c18fcf8f24c5c14e1/LICENSE).

## Download and load

The first fit downloads and caches the files for the estimator's task. The
checkpoints are public and do not require signing in. Construction and
scikit-learn cloning defer weight loading until fit.

For a manual download, use the revision listed for your task. Keep `config.json`
beside `model.safetensors` and pass the directory or weights file as
`checkpoint_path`. Download both files from the same revision. The package
checks that the weights match the configuration and prediction task before
loading the model.

```python
from causilo import CausiloClassifier, CausiloRegressor

regressor = CausiloRegressor(checkpoint_path="checkpoints/regressor")
classifier = CausiloClassifier(checkpoint_path="checkpoints/classifier", mode="light")
```

The 1.5 `config.json` identifies the architecture, checkpoint format,
configuration version, task and model dimensions. Its exact file bytes are
bound to `model.safetensors` through the `config_sha256` metadata field.

## File checksums

SHA-256 for the release files at the pinned revisions above. The
[benchmark details](benchmarks/README.md) record the checksums used for evaluation.

| Checkpoint | `config.json` | `model.safetensors` |
| --- | --- | --- |
| 1.5 classification | `900b2bbbab73ae8e681bab2e418ce8a38143785706e092505b691544cea68764` | `b378d856b0d4cf894426b5d2023db186c44b5e0d1b9261046ccb1f433bc99bf2` |
| 1.5 regression | `489670f1c479fa4aa49e77cf1080ae9f8899cad1ecf9dac3215dca77f243da97` | `0f4e40f1ca671dd5d85cbb159e7e638606274f02bc73cdf861265a10fedbf3be` |
| 1.0 classification | `89aa44afe34233013f5c38a7dd89f233d26d18585096fcf32dec250abd48b77d` | `2fe50487b33ef6b4ed65d1dab1e48888b86f9ed604317a4b596fea4dd462b7f6` |
| 1.0 regression | `af78b2d23ac28534ab6c16f75d8d2d2f4dc38ca45c643d70a6324320847480a7` | `9a4197d7c1060d187c19b62d4d83c74ce986c1f3911825576eb58c2ae4f0057d` |
