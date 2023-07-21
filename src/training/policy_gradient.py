# train Policy Gradient with Advantage Actor Critic algorithm

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from open_spiel.python import policy, rl_environment
from open_spiel.python.algorithms import policy_gradient, exploitability
import time

from src.shared_flags import *
from src.methods import *

FLAGS = flags.FLAGS

flags.DEFINE_enum("training_agent", "PolicyGradient", VALID_PLAYERS, "Name of Agent to train")

# a2c = Advantage Actor Critic
# rpg = Regret Policy Gradient
# qpg = Q-based Policy Gradient
# rm  = Regret Matching Policy Gradient
flags.DEFINE_enum("loss_algorithm", "rpg", ["a2c", "rpg", "qpg", "rm"], "Algorithm for loss to use.")


class PolicyGradientPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies):
    game = env.game
    player_ids = [0, 1]
    super(PolicyGradientPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(_):

  print("-------- TRAIN {} --------".format(FLAGS.training_agent+"_"+FLAGS.loss_algorithm))

  # import and init Weights & Biases Service
  if FLAGS.wandb_enable:
    import wandb, random
    wandb_run_name="{} {} {}".format(FLAGS.training_agent+"_"+FLAGS.loss_algorithm, FLAGS.game, str(random.randrange(1000)))
    print("WANDB RUN NAME: {}".format(wandb_run_name))
    wandb.init(
      project=FLAGS.wandb_project_name,
      tags=[FLAGS.game, FLAGS.training_agent],
      group=FLAGS.training_agent,
      name=wandb_run_name
    )
    wandb.config.update(FLAGS)

  game = FLAGS.game
  num_players = FLAGS.game_number_players

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  new_path_checkpoint = create_checkpoint_path_from_arguments(FLAGS.game, FLAGS.training_agent+"_"+FLAGS.loss_algorithm, FLAGS.save_checkpoint_dir, is_nn_model=True)



  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        policy_gradient.PolicyGradient(
            sess,
            idx,
            info_state_size,
            num_actions,
            loss_str=FLAGS.loss_algorithm,
            hidden_layers_sizes=(128,)) for idx in range(num_players)
    ]
    expl_policies_avg = PolicyGradientPolicies(env, agents)

    # load trained model
    if FLAGS.load_enable:
        for agent in agents:
          if agent.has_checkpoint(new_path_checkpoint):
            print("LOADING MODEL ...")
            agent.restore(new_path_checkpoint)
    else:
      sess.run(tf.global_variables_initializer())

    start_time = time.time()
    nash_conv=-1
    nash_conv_before=-100
    for ep in range(FLAGS.training_number_episodes):

      if (ep + 1) % FLAGS.training_evaluation_frequency == 0:
        losses = [agent.loss for agent in agents]
        time_done = time.time() - start_time
        nash_conv_before=nash_conv
        nash_conv = exploitability.nash_conv(env.game, expl_policies_avg)
        msg = "-" * 80 + "\n"
        msg += "{}: {}\n{}\n".format(ep + 1, nash_conv, losses)
        logging.info("%s", msg)
        print("Time so far: {}".format(time_done))

        # log to Weights & Biases Service
        time_done = time.time() - start_time
        if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "Iteration": ep, "Zeit": time_done}, commit=True)

          # save agent
        if (ep + 1) % FLAGS.save_frequency == 0:
          if FLAGS.save_enable:
            for agent in agents:
              agent.save(new_path_checkpoint)

      else:
        if (ep + 1) % FLAGS.training_log_frequency == 0:
          time_done = time.time() - start_time
          if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "Iteration": ep, "Zeit": time_done}, commit=False)



        # exit training if good enough
      if is_nash_convergence_good_enough(nash_conv_now=nash_conv, nash_conv_before=nash_conv_before , time_training_in_seconds=(time.time() - start_time), end_after_n_hours=6):

        # log to Weights & Biases Service
        time_done = time.time() - start_time
        if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "Iteration": ep, "Zeit": time_done}, commit=True)
        if FLAGS.save_enable:
            for agent in agents:
              agent.save(new_path_checkpoint)
        exit()

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
