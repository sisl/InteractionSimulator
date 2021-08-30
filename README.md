# InteractionSimulator
Simulator for the INTERACTION dataset

## Installation

### Dependencies

Install requirements with pip

```
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Dataset

The INTERACTION dataset contains a two folders which should be copied into the directory that the `INTERSIM_DATSET_DIR` environment variable points to:
  - the contents of ``recorded_trackfiles`` should be copied to ``$INTERSIM_DATASET_DIR/trackfiles``
  - the contents of ``maps`` should be copied to ``$INTERSIM_DATASET_DIR/maps``
If the `INTERSIM_DATSET_DIR` environment variable is not set, it will default to the `./datasets` directory (i.e. at the repo root).

### Tests

``python tests/test_idm_graph.py`` should generate a 300-timeframe-long simulation video, `idm_graph.mp4`, with an IDM policy and ClosestObstacle graph.
