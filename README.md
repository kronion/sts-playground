STS Playground
---

A project for experimentation on the Slay the Spire Gym env.

## Installation

1. You will need a copy of Slay the Spire as well as several mods in order to use the
   `gym-sts` Gym environment. See [the project's GitHub](https://github.com/kronion/gym-sts/)
   for instructions. How specifically you set up your lib and mods is up to you, but this is
   on recommended setup:
   ```
   sts-playground/
       lib/
           desktop-1.0.jar
           ModTheSpire.jar
       mods/
           BaseMod.jar
           CommunicationModFork.jar
           SuperFastModeFork.jar
   ```

2. Make sure you have Java JDK 8 installed. On Ubuntu you can install it like this:
   ```zsh
   [sudo] apt install openjdk-8-jdk
   ```
   If you have multiple JDKs installed, you may need to select JDK 8 to be used as the default:
   ```
   [sudo] update-alternatives --config java
   ```

3. If you don't have Steam installed on your machine, you need to make sure ModTheSpire finds
   the correct JRE. Add a symlink to the correct java binary in your lib directory, something like this:
   ```zsh
   # within your lib dir
   mkdir -p jre/bin
   ln -s java jre/bin/java
   ```

4. Install Python dependencies. This project uses [poetry](https://python-poetry.org/) for dependency management.
   Simply `poetry install` and `poetry shell`.

5. If you want to run multiple copies of the game simultaneously for faster training, you can run the
   env instances in "headless" mode. You will need to have `docker` installed. Each game instance will run
   in a separate container. Note that you will not be able to observe agent play when running headlessly.

6. This project uses tensorflow as the framework for defining neural nets. GPU acceleration is supported, but
   you will need to [follow the tensorflow installation instructions carefully](https://www.tensorflow.org/install/pip).
   In particular, we expect that you will have cudatoolkit and CUDNN installed in a way that tensorflow is happy with.
   Using `conda` to install `cudatoolkit` works well, and we include the `CUDNN` library as a poetry dependency (but
   note that you'll still need to set various environment variables for tensorflow to find and use the library). Also
   note that tensorflow has additional instructions for Ubuntu 22.04.

   If you install Python dependencies with `poetry`, and `cudatoolkit` with `conda`, then the following commands should
   be all you need to set up your environment:
   ```zsh
   # Enter your Python virtualenv
   poetry shell

   # Add system dependencies installed with conda.
   # NB: You should NOT have a Python installed in your conda env, unless you know what you're doing.
   # Otherwise, you might clobber the Python environment managed by poetry.
   conda activate [your-env]

   # If you don't automatically set environment vars when entering your conda env,
   # as suggested in the tensorflow docs, you should set them now.
   ```

## Running experiments

This project uses [Poe the Poet](https://github.com/nat-n/poethepoet) to define reusable
executable commands, like you would with a Makefile. If you've installed the project with
`poetry`, you should be able to use them automatically. You're welcome to run the
underlying commands yourself, though. You can find them in the `[tool.poe.tasks]` section
of the `pyproject.toml` file. These instructions assume you're using `poe`.


## RL Model Training

```zsh
poe rl_train
```

List available CLI flags:
```
poe rl_train --help
```

## Autoencoder Training

```zsh
poe autoencoder_train --data data/states_relic_tuple.pkl --batch_size=200
```

List available CLI flags:
```
poe autoencoder_train --help
```

