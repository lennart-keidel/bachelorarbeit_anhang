""" train NFSP, Neural Fictitous Self Play """

from absl import app
from absl import flags
from absl import logging
import time
import tensorflow.compat.v1 as tf

from open_spiel.python import policy, rl_environment
from open_spiel.python.algorithms import nfsp, exploitability

from src.shared_flags import *
from src.methods import *

FLAGS = flags.FLAGS

flags.DEFINE_enum("training_agent", "NFSP", VALID_PLAYERS, "Name of Agent to train")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5), "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6), "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000, "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1, "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128, "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64, "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01, "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01, "Learning rate for avg-policy sl network.")
flags.DEFINE_string("optimizer_str", "sgd", "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse", "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 19200, "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0, "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6), "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.06, "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.001, "Final exploration parameter.")


class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = list(range(FLAGS.game_number_players))
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {
        "info_state": [None] * FLAGS.game_number_players,
        "legal_actions": [None] * FLAGS.game_number_players
    }

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(unused_argv):

  # import and init Weights & Biases Service
  if FLAGS.wandb_enable:
    import wandb, random
    wandb_run_name="{} {} {}".format(FLAGS.training_agent, FLAGS.game, str(random.randrange(1000)))
    print("WANDB RUN NAME: {}".format(wandb_run_name))
    wandb.init(
      project=FLAGS.wandb_project_name,
      tags=[FLAGS.game, FLAGS.training_agent],
      group=FLAGS.training_agent,
      name=wandb_run_name
    )
    wandb.config.update(FLAGS)

  print("Loading %s", FLAGS.game)
  game = FLAGS.game
  num_players = FLAGS.game_number_players

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "reservoir_buffer_capacity": FLAGS.reservoir_buffer_capacity,
      "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
      "anticipatory_param": FLAGS.anticipatory_param,
      "batch_size": FLAGS.batch_size,
      "learn_every": FLAGS.learn_every,
      "rl_learning_rate": FLAGS.rl_learning_rate,
      "sl_learning_rate": FLAGS.sl_learning_rate,
      "optimizer_str": FLAGS.optimizer_str,
      "loss_str": FLAGS.loss_str,
      "update_target_network_every": FLAGS.update_target_network_every,
      "discount_factor": FLAGS.discount_factor,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "epsilon_start": FLAGS.epsilon_start,
      "epsilon_end": FLAGS.epsilon_end,
  }

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  **kwargs) for idx in range(num_players)
    ]
    joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
    new_path_checkpoint = create_checkpoint_path_from_arguments(FLAGS.game, "NFSP", FLAGS.save_checkpoint_dir, is_nn_model=True)
    sess.run(tf.global_variables_initializer())

    if FLAGS.load_enable:
      for agent in agents:
        if agent.has_checkpoint(new_path_checkpoint):
          print("LOADING MODEL ...")
          agent.restore(new_path_checkpoint)

    start_time = time.time()
    nash_conv=-1
    nash_conv_before=-100
    for ep in range(FLAGS.training_number_episodes):
      if (ep + 1) % FLAGS.training_evaluation_frequency == 0:

        losses = [agent.loss for agent in agents]
        time_done = time.time() - start_time
        nash_conv = exploitability.nash_conv(env.game, joint_avg_policy)

        # log
        print("-------- TRAIN NFSP ITERATION {} --------".format(ep))
        print("Losses: %s", losses)
        print("[%s] NashConv %s", ep + 1, nash_conv)
        print("Time so far: {}".format(time_done))


        # save model
        if (ep + 1) % FLAGS.save_frequency == 0:
          # log to Weights & Biases Service
          if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "Iteration": ep, "Zeit": time_done}, commit=True)

          if FLAGS.save_enable:
            print("SAVING MODEL ...")
            for agent in agents:
              agent.save(new_path_checkpoint)

          # exit training if good enough
          if is_nash_convergence_good_enough(nash_conv_now=nash_conv, nash_conv_before=nash_conv_before , time_training_in_seconds=(time.time() - start_time), end_after_n_hours=6):
            print("SAVING MODEL ...")
            for agent in agents:
              agent.save(new_path_checkpoint)
            exit()
          nash_conv_before=nash_conv
        else:
          if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "Iteration": ep, "Zeit": time_done}, commit=False)
        print("_____________________________________________")

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
