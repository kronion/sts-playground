import getpass

from absl import app, flags
import fancyflags as ff
import wandb

from sts_playground.autoencoder import tf_lib

DATA = flags.DEFINE_string("data", "data/state_action_triples_100k.pkl", "path to data")

LR = flags.DEFINE_float("lr", 1e-4, "learning rate")
BATCH_SIZE = flags.DEFINE_integer("batch_size", 1024, "batch size")
LOSS = flags.DEFINE_enum("loss", "ce", ["ce", "mse"], "type of loss")

NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 10, 'number of training epochs')
COMPILE = flags.DEFINE_boolean('compile', True, 'Use tf.function.')
DATA_LIMIT = flags.DEFINE_integer('data_limit', None, 'Limit dataset size to save memory.')

NETWORK = ff.DEFINE_dict(
    'network',
    depth=ff.Integer(1, 'number of intermediate layers'),
    width=ff.Integer(128, 'size of each intermediate layer'),
)

# passed to wandb.init
WANDB = ff.DEFINE_dict(
    'wandb',
    entity=ff.String('sts-ai'),
    project=ff.String('next-state-prediction'),
    mode=ff.Enum('online', ['online', 'offline', 'disabled']),
    group=ff.String(getpass.getuser()),  # group by username
    name=ff.String(None),
    notes=ff.String(None),
)


def main(_):
    wandb.init(
        config=dict(
            lr=LR.value,
            batch_size=BATCH_SIZE.value,
            loss=LOSS.value,
            network=NETWORK.value,
            data_limit=DATA_LIMIT.value,
        ),
        **WANDB.value,
    )

    tf_lib.train(
        data_path=DATA.value,
        batch_size=BATCH_SIZE.value,
        network_config=NETWORK.value,
        learning_rate=LR.value,
        num_epochs=NUM_EPOCHS.value,
        logger=wandb.log,
        loss_type=LOSS.value,
        compile=COMPILE.value,
        data_limit=DATA_LIMIT.value,
    )

if __name__ == "__main__":
    app.run(main)
