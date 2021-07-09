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

The INTERACTION dataset contains a two folders which should be copied into a folder called ``./datasets``: 
  - the contents of ``recorded_trackfiles`` should be copied to ``./datasets/trackfiles``
  - the contents of ``maps`` should be copied to ``./datasets/maps``

### Tests

``python tests/test_simulator.py`` should generate a 10-timeframe-long simulation video, `tests/output/test_simulator_ani.mp4`, with randomly sampled actions from the action space.

If issues with ffmpeg and using Conda to manage environment, can resolve with ``conda install -c conda-forge ffmpeg``
