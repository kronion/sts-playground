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

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': {
        'batch_size': {'values': [256, 512, 1024]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 1e-3, 'min': 1e-5, 'distribution': 'log_uniform_values'},
        # 'network': {
        #     'depth': {'values': [1, 2, 4]},
        #     'width': {'values': [128, 256, 512]},
        # },
    },
}


# passed to wandb.init
WANDB = ff.DEFINE_dict(
    'wandb',
    entity=ff.String('sts-ai'),
    project=ff.String('autoencoder'),
    group=ff.String(getpass.getuser()),  # group by username
    name=ff.String(None),
    notes=ff.String(None),
)

def train():
    wandb.init(
        config=dict(
            data=DATA.value,
            data_limit=DATA_LIMIT.value,
            network=NETWORK.value,
            compile=COMPILE.value,
            batch_size=BATCH_SIZE.value,
            epochs=NUM_EPOCHS.value,
            lr=LR.value,
            loss=LOSS.value,
        ),
        entity='sts-ai',
        project='autoencoder',
        group=getpass.getuser(),
        # TODO: make name and notes configurable
    )

    tf_lib.train(
        data_path=wandb.config.data,
        batch_size=wandb.config.batch_size,
        network_config=wandb.config.network,
        learning_rate=wandb.config.lr,
        num_epochs=wandb.config.epochs,
        loss_type=wandb.config.loss,
        compile=wandb.config.compile,
        data_limit=wandb.config.data_limit,
        logger=wandb.log,
    )

def main(_):
    sweep_id = wandb.sweep(sweep=sweep_configuration)
    wandb.agent(sweep_id, function=train, count=4)

if __name__ == "__main__":
    app.run(main)
