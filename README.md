STS Playground
---

A project for experimentation on the Slay the Spire Gym env.

## Installation

This project uses `poetry` for dependency management. Simply `poetry install` and
`poetry shell`.

You will need a copy of Slay the Spire as well as several mods in order to use the
`gym-sts` Gym environment. See [the project's GitHub](https://github.com/kronion/gym-sts/)
for instructions.

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

