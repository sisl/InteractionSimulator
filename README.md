# InteractionSimulator
Simulator for the INTERACTION dataset

## Installation

### Dependencies

Install requirements with pip

```
pip install -r requirements.txt
pip install -e .
```

### Dataset

The INTERACTION dataset contains a two folders which should be copied into the directory that the `INTERSIM_DATSET_DIR` environment variable points to:
  - the contents of ``recorded_trackfiles`` should be copied to ``$INTERSIM_DATASET_DIR/trackfiles``
  - the contents of ``maps`` should be copied to ``$INTERSIM_DATASET_DIR/maps``
If the `INTERSIM_DATSET_DIR` environment variable is not set, it will default to the `./datasets` directory (i.e. at the repo root).

### Tests

``python test_scripts/test_simulator.py`` should generate a 10-timeframe-long simulation video, `tests/output/test_simulator_ani.mp4`, with randomly sampled actions from the action space.

If issues with ffmpeg and using Conda to manage environment, can resolve with ``conda install -c conda-forge ffmpeg``
