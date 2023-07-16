from typing import Dict

from gym_sts.spaces import actions
from gym_sts.spaces.observations import Observation
from gym_sts.spaces.constants.base import ScreenType
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomMetricCallbacks(DefaultCallbacks):
    # def on_episode_start(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: Episode,
    #     env_index: int,
    #     **kwargs
    # ):
    #     # Make sure this episode has just been started (only initial obs
    #     # logged so far).
    #     assert episode.length == 0, (
    #         "ERROR: `on_episode_start()` callback should be called right "
    #         "after env reset!"
    #     )
    #     # print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
    #     episode.user_data["pole_angles"] = []
    #     # episode.hist_data["pole_angles"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

        raw_obs = episode.last_raw_obs_for()
        if raw_obs is None:
            return
        obs = Observation.deserialize(raw_obs)

        data = episode.user_data

        if "campfire_options" in data:
            raw_action = episode.last_action_for()
            action = actions.ACTIONS[raw_action]

            if isinstance(action, actions.Choose):
                choices = data["campfire_options"]
                choice = choices[action.choice_index]

                if "rest_choices" not in data:
                    data["rest_choices"] = {}

                rest_choices = data["rest_choices"]
                if choice not in rest_choices:
                    rest_choices[choice] = 0
                rest_choices[choice] += 1

                if "total_rests" not in data:
                    data["total_rests"] = 0
                data["total_rests"] += 1

                del data["campfire_options"]

        elif obs.persistent_state.screen_type == ScreenType.REST:
            # if "here" not in episode.user_data:
            #     episode.user_data["here"] = 0
            # episode.user_data["here"] += 1

            if not obs.campfire_state.has_rested:
                data["campfire_options"] = obs.campfire_state.options

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        total_rests = episode.user_data.get("total_rests", 0)
        episode.custom_metrics["total_rests"] = total_rests

        if total_rests > 0:
            rest_choices = episode.user_data["rest_choices"]
            for choice, count in rest_choices.items():
                episode.custom_metrics[choice.name] = count / total_rests
