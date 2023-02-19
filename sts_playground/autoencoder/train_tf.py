import pickle
import getpass
import time
import typing as tp

from absl import app, flags
import fancyflags as ff
import numpy as np
import tensorflow as tf
import sonnet as snt
import tree
from gym import spaces
import wandb

from gym_sts.envs import base
from sts_playground import utils

DATA = flags.DEFINE_string("data", "data/states.pkl", "path to data")

LR = flags.DEFINE_float("lr", 1e-4, "learning rate")
BATCH_SIZE = flags.DEFINE_integer("batch_size", 100, "batch size")
LOSS = flags.DEFINE_enum("loss", "ce", ["ce", "mse"], "type of loss")

NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 10, 'number of training epochs')
COMPILE = flags.DEFINE_boolean('compile', True, 'Use tf.function.')

NETWORK = ff.DEFINE_dict(
    'network',
    depth=ff.Integer(1, 'number of intermediate layers'),
    width=ff.Integer(128, 'size of each intermediate layer'),
)

# passed to wandb.init
WANDB = ff.DEFINE_dict(
    'wandb',
    entity=ff.String('sts-ai'),
    project=ff.String('autoencoder'),
    mode=ff.Enum('offline', ['online', 'offline', 'disabled']),
    group=ff.String(getpass.getuser()),  # group by username
    name=ff.String(None),
    notes=ff.String(None),
)

def space_as_nest(space: spaces.Space):
    if isinstance(space, spaces.Dict):
        return {k: space_as_nest(v) for k, v in space.items()}
    if isinstance(space, spaces.Tuple):
        return tuple(map(space_as_nest, space))
    return space


def encode(space, x) -> tf.Tensor:
    """Gives a flat encoding of a structured observation."""
    if isinstance(space, spaces.Discrete):
        return tf.one_hot(x, space.n)
    if isinstance(space, spaces.MultiBinary):
        return tf.cast(x, tf.float32)
    if isinstance(space, spaces.MultiDiscrete):
        one_hots = []
        for n, y in zip(space.nvec, tf.unstack(x, axis=-1)):
            one_hots.append(tf.one_hot(y, n))
        return tf.concat(one_hots, axis=-1)
    if isinstance(space, (spaces.Dict, spaces.Tuple)):
        encodings = tree.map_structure(
            encode, space_as_nest(space), x, check_types=False
        )
        return tf.concat(tree.flatten(encodings), axis=-1)
    raise NotImplementedError(type(space))


def space_size(space) -> int:
    if isinstance(space, spaces.Discrete):
        return space.n
    if isinstance(space, spaces.MultiBinary):
        assert isinstance(space.n, int)
        return space.n
    if isinstance(space, spaces.MultiDiscrete):
        return space.nvec.sum()
    if isinstance(space, spaces.Dict):
        space = space_as_nest(space)
        return sum(map(space_size, tree.flatten(space)))
    raise NotImplementedError(type(space))


def to_tensor(x: np.ndarray) -> tf.Tensor:
    if x.dtype == np.uint16:
        x = x.astype(np.int16)
    return tf.constant(x)

def ce_loss(x, y):
  y = tf.one_hot(y, depth=x.shape[-1])
  return tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x)

def struct_loss(space: spaces.Space, x: tf.Tensor, y: tf.Tensor):
    if isinstance(space, spaces.Discrete):
        return ce_loss(x, y)

    if isinstance(space, spaces.MultiBinary):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(y, tf.float32), logits=x)
        return tf.reduce_sum(loss, axis=-1)

    if isinstance(space, spaces.MultiDiscrete):
        num_components = space.nvec.shape[0]
        assert y.shape[-1] == num_components
        assert x.shape[-1] == space.nvec.sum()

        n = space.nvec[0]
        if np.all(space.nvec == n):
          # If all the Discrete's are the same size, we can avoid the expensive
          # unstack and do a reshape instead.
          old_shape = x.shape.as_list()
          new_shape = old_shape[:-1] + [num_components, n]
          new_x = tf.reshape(x, new_shape)
          losses = ce_loss(new_x, y)
          return tf.reduce_sum(losses, axis=-1)

        losses = []
        logits = tf.split(x, tuple(space.nvec), axis=-1)
        for x_, y_ in zip(logits, tf.unstack(y, axis=-1)):
            losses.append(ce_loss(x_, y_))
        return tf.add_n(losses)

    if isinstance(space, (spaces.Dict, spaces.Tuple)):
        space = space_as_nest(space)
        flat_spaces = tree.flatten(space)
        space_sizes = tuple(map(space_size, flat_spaces))
        assert sum(space_sizes) == x.shape[-1]
        xs = tf.split(x, space_sizes, axis=-1)
        xs_tree = tree.unflatten_as(y, xs)
        return tree.map_structure(struct_loss, space, xs_tree, y, check_types=False)

    raise NotImplementedError(type(space))


def discrete_accuracy(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    argmax = tf.argmax(x, axis=-1)
    y = tf.cast(y, argmax.dtype)
    return tf.cast(tf.equal(argmax, y), tf.float32)


def accuracy(space: spaces.Space, x: tf.Tensor, y: tf.Tensor):
    if isinstance(space, spaces.Discrete):
        return discrete_accuracy(x, y)

    if isinstance(space, spaces.MultiBinary):
        assert x.shape[-1] == space.n
        assert x.shape == y.shape
        prediction = tf.cast(tf.greater(x, 0), y.dtype)
        correct = tf.equal(prediction, y)
        return tf.reduce_mean(tf.cast(correct, tf.float32), axis=-1)

    if isinstance(space, spaces.MultiDiscrete):
        num_components = space.nvec.shape[0]
        assert y.shape[-1] == num_components
        assert x.shape[-1] == space.nvec.sum()

        n = space.nvec[0]
        if np.all(space.nvec == n):
          # If all the Discrete's are the same size, we can avoid the expensive
          # unstack and do a reshape instead.
          old_shape = x.shape.as_list()
          new_shape = old_shape[:-1] + [num_components, n]
          new_x = tf.reshape(x, new_shape)
          losses = discrete_accuracy(new_x, y)
          return tf.reduce_mean(losses, axis=-1)

        losses = []
        logits = tf.split(x, tuple(space.nvec), axis=-1)
        for x_, y_ in zip(logits, tf.unstack(y, axis=-1)):
            losses.append(discrete_accuracy(x_, y_))
        loss = tf.stack(losses, axis=-1)
        return tf.reduce_mean(loss, axis=-1)

    if isinstance(space, (spaces.Dict, spaces.Tuple)):
        space = space_as_nest(space)
        flat_spaces = tree.flatten(space)
        space_sizes = tuple(map(space_size, flat_spaces))
        assert sum(space_sizes) == x.shape[-1]
        xs = tf.split(x, space_sizes, axis=-1)
        xs_tree = tree.unflatten_as(y, xs)
        return tree.map_structure(accuracy, space, xs_tree, y, check_types=False)

    raise NotImplementedError(type(space))


def make_auto_encoder(input_size: int, depth: int, width: int) -> snt.Module:
    return snt.nets.MLP(
        output_sizes=[width] * depth + [input_size],
        activation=tf.nn.leaky_relu,
    )

def fix_dtype(x: np.ndarray):
  """Fix types that tensorflow can't handle properly."""
  if x.dtype in (np.uint16,):
    return x.astype(np.int32)
  return x

def convert_lists(struct):
    """Convert all lists to tuples in a struct."""
    if isinstance(struct, tp.Mapping):
        return {k: convert_lists(v) for k, v in struct.items()}
    if isinstance(struct, tp.Sequence):
        return tuple(map(convert_lists, struct))
    return struct

def main(_):
    # download from https://drive.google.com/file/d/1R5eyUebTXsNJV4PCoDQE9lWfKxlWmsNs/view?usp=share_link  # noqa: E501
    with open(DATA.value, "rb") as f:
        column_major = pickle.load(f)
    
    column_major = column_major['state_before']
    column_major = tree.map_structure(fix_dtype, column_major)
    column_major = convert_lists(column_major)

    obs_space = space_as_nest(base.OBSERVATION_SPACE)
    for diff in utils.tree_diff(obs_space, column_major):
        print(diff)
        import ipdb; ipdb.set_trace()
    tree.assert_same_structure(obs_space, column_major)

    total_size = len(tree.flatten(column_major)[0])
    train_size = round(total_size * 0.8)
    print("total states:", total_size)

    wandb.init(
        config=dict(
            lr=LR.value,
            batch_size=BATCH_SIZE.value,
            loss=LOSS.value,
            dataset_size=total_size,
        ),
        **WANDB.value,
    )

    # import tensorflow.python.data.util as util
    # util.structure.type_spec_from_value(column_major)

    train_set = tree.map_structure(lambda x: x[:train_size], column_major)
    valid_set = tree.map_structure(lambda x: x[train_size:], column_major)

    # Shuffle the train set in numpy because tfds is very slow.
    rng = np.random.RandomState(0)
    tree.map_structure(rng.shuffle, train_set)

    # train_tensor = tree.map_structure(to_tensor, train_set)
    # valid_tensor = tree.map_structure(to_tensor, valid_set)

    input_size = space_size(base.OBSERVATION_SPACE)
    print("input_size", input_size)

    flat_spaces = tree.flatten(obs_space)
    print("num components:", len(flat_spaces))

    # prepare data
    # flat_train_set = tuple(tree.flatten(train_set))
    # We could avoid flattening if we used tuples instead of lists in obs struct.
    train_ds = tf.data.Dataset.from_tensor_slices(train_set)
    assert len(train_ds) == train_size
    # train_ds = train_ds.shuffle(train_size, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(BATCH_SIZE.value, drop_remainder=True)

    input_dim = space_size(base.OBSERVATION_SPACE)
    auto_encoder = make_auto_encoder(input_dim, **NETWORK.value)

    optimizer = snt.optimizers.Adam(LR.value)

    def compute_loss(batch) -> tp.Tuple[tf.Tensor, dict]:
        metrics = {}

        flat_input = encode(base.OBSERVATION_SPACE, batch)
        flat_output = auto_encoder(flat_input)

        if LOSS.value == "ce":
            losses = struct_loss(base.OBSERVATION_SPACE, flat_output, batch)
            metrics['ce'] = tree.map_structure(
                lambda x: tf.reduce_mean(x, axis=0),
                losses)
            loss = tf.add_n(tree.flatten(losses))  # [B]
        elif LOSS.value == "mse":
            losses = tf.square(flat_input - flat_output)  # [B, D]
            loss = tf.reduce_sum(losses, axis=1)  # [B]
        loss = tf.reduce_mean(loss, axis=0)  # []
        metrics['loss'] = loss

        top1 = accuracy(base.OBSERVATION_SPACE, flat_output, batch) #  {[B]}
        top1 = tree.map_structure(  # {[]}
            lambda x: tf.reduce_mean(x, axis=0), top1)
        top1_mean = tf.reduce_mean(tf.stack(tree.flatten(top1)))  # []
        
        metrics['top1'] = top1
        metrics['top1_mean'] = top1_mean

        return loss, metrics

    # TODO: reduce compilation memory/overhead and re-enable compilation here.
    # if COMPILE.value:
    #   compute_loss = tf.function(compute_loss)

    def train_step(batch) -> dict:
      with tf.GradientTape() as tape:
        loss, metrics = compute_loss(batch)
      
      params: list[tf.Variable] = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in auto_encoder.trainable_variables]
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(loss, params)
      optimizer.apply(grads, params)

      return metrics

    if COMPILE.value:
      train_step = tf.function(train_step, autograph=False)

    def print_results(results: dict, prefix: str = ""):
        loss = results["loss"]
        top1 = results["top1_mean"]
        print(f"{prefix} loss={loss:.1f} top1={top1:.5f}", end="\n")

    step = 0
    for epoch in range(NUM_EPOCHS.value):
        print(f"Epoch {epoch}")
        total_batches = len(train_ds)
        for batch_num, batch in enumerate(train_ds):
            start_time = time.perf_counter()
            results = train_step(batch)
            step_time = time.perf_counter() - start_time

            loss = results["loss"]
            top1 = results["top1_mean"]
            print(
               f"Batch: {batch_num+1}/{total_batches} "
               f"loss={loss:.1f} top1={top1:.5f} time={step_time:.2f}s",
               end="\n")

            to_log = dict(
                loss=results["loss"],
                top1=results["top1"],
                epoch=epoch + batch_num / total_batches,
            )
            wandb.log(dict(train=to_log), step=step)

            step += 1

        # TODO: we should split up the validation set into batches
        _, val_results = compute_loss(valid_set)
        print_results(val_results, "Validation:")
        # TODO: use validation loss to adjust learning rate

        to_log = dict(
            loss=val_results["loss"],
            top1=val_results["top1"],
            epoch=epoch + 1,
        )
        wandb.log(dict(validation=to_log), step=step)


if __name__ == "__main__":
    app.run(main)
