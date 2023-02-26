import pickle
import getpass

from absl import app, flags
import fancyflags as ff
import numpy as np
import tree
import wandb

from sts_playground.autoencoder import tf_lib

DATA = flags.DEFINE_string("data", "data/states.pkl", "path to data")

LR = flags.DEFINE_float("lr", 1e-4, "learning rate")
BATCH_SIZE = flags.DEFINE_integer("batch_size", 1024, "batch size")
LOSS = flags.DEFINE_enum("loss", "ce", ["ce", "mse"], "type of loss")

NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 10, 'number of training epochs')
COMPILE = flags.DEFINE_boolean('compile', True, 'Use tf.function.')

DATA_LIMIT = flags.DEFINE_integer('data_limit', None, 'limit the dataset size')

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
    mode=ff.Enum('offline', ['online', 'offline', 'disabled']),
    group=ff.String(getpass.getuser()),  # group by username
    name=ff.String(None),
    notes=ff.String(None),
)


def main(_):
    # download from https://drive.google.com/file/d/180068R95gdt-OAMm4-79bTlB2uq4P1UX/view?usp=share_link  # noqa: E501
    with open(DATA.value, "rb") as f:
        data = pickle.load(f)
    data['action'] = np.array(data['action'])

    dataset_size = len(data['action'])
    data_limit = DATA_LIMIT.value
    if data_limit:
        if data_limit > dataset_size:
            raise ValueError('Data limit {data_limit} exceeds dataset size {dataset_size}.')
        data = tree.map_structure(lambda xs: xs[:data_limit].copy(), data)
        print(f'Limited dataset size to {data_limit}')
        dataset_size = data_limit

    # preprocess the data
    data = tree.map_structure(tf_lib.fix_dtype, data)
    data = tf_lib.convert_lists(data)

    wandb.init(
        config=dict(
            lr=LR.value,
            batch_size=BATCH_SIZE.value,
            loss=LOSS.value,
            network=NETWORK.value,
            dataset_size=dataset_size,
        ),
        **WANDB.value,
    )

    tf_lib.train(
        column_major=data,
        batch_size=BATCH_SIZE.value,
        network_config=NETWORK.value,
        learning_rate=LR.value,
        num_epochs=NUM_EPOCHS.value,
        logger=wandb.log,
        loss_type=LOSS.value,
        compile=COMPILE.value,
    )

if __name__ == "__main__":
    app.run(main)
