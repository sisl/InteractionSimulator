# InteractionSimulator
OpenAI gym simulator for the [INTERACTION dataset](https://interaction-dataset.com/)

## Installation

### Dataset

To use this simulator, first download the INTERACTION dataset (above). The dataset contains two folders which should be copied into a folder called ``./datasets``: 
  - the contents of ``recorded_trackfiles`` should be copied to ``./datasets/trackfiles``
  - the contents of ``maps`` should be copied to ``./datasets/maps``

### Dependencies

Install requirements and build environment with pip:

```
pip install -r requirements.txt
pip install -e .
```

### Tests

To ensure correct installation, tests can be run with ``pytest``.

If you have issues with ffmpeg and are using Conda to manage environment, you can resolve this with ``conda install -c conda-forge ffmpeg``

## Environments

### Intersim

The ``intersim-v0`` gym environment loads a track and map file from the dataset. It sets the vehicle tracks and spawn times from the dataset, estimating tracks with a 20th order polynomial. The user specifies a single acceleration action for each vehicle present in the scene, and the vehicles follow double integrator dynamics along their tracks. The environment supports customizable observations, as well as an interaction graph class in order to mask the observations based on which vehicles are 'interacting'. This environment is appropriate for multi-agent decision making. For details, see ``intersim/envs/simulator.py``.

An intersim environment can be initialized with: 
```
import gym
env=gym.make('intersim:intersim-v0')
```

### Intersimple

The ``intersimple`` environment classes build on top of ``intersim-v0`` to allow for the control of a single agent while the remaining agents follow their estimated accelerations (and therefore recorded tracks) from the dataset. Accelerations are estimated from the positions in the datasets while using simple L2 smoother. For details, see ``intersim/envs/intersimple.py``.

### Rendering

Using ``env.render()`` will add the environment states and actions to a replay buffer. ``env.close(filestr='output/render')`` will then animate the buffer and save the video, along with all relevant data to a file location with base string ``'output/render_'``.