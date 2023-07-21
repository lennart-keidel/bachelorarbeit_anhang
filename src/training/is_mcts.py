""" train information set monte-carlo-tree-search """

from absl import app
from absl import flags
import time

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import pyspiel

from src.shared_flags import *
from src.methods import *

FLAGS = flags.FLAGS


flags.DEFINE_enum("training_agent", "ISMCTS", VALID_PLAYERS, "Name of Agent to train")

SEED = 981095111


def construct_is_mcts_policy(game, state, tabular_policy, bot, searched):
  """Constructs a tabular policy from independent bot calls.

  Args:
    game: an OpenSpiel game,
    state: an OpenSpiel state to start the tree walk from,
    tabular_policy: a policy.TabularPolicy for this game,
    bot: the bot to get the policy from at each state
    searched: a dictionary of information states already search (empty to begin)
  """

  if state.is_terminal():
    return
  elif state.is_chance_node():
    outcomes = state.legal_actions()
    for outcome in outcomes:
      new_state = state.clone()
      new_state.apply_action(outcome)
      construct_is_mcts_policy(game, new_state, tabular_policy, bot, searched)
  else:
    infostate_key = state.information_state_string()
    if infostate_key not in searched:
      searched[infostate_key] = True
      infostate_policy = bot.get_policy(state)
      tabular_state_policy = tabular_policy.policy_for_key(infostate_key)
      for action, prob in infostate_policy:
        tabular_state_policy[action] = prob
    for action in state.legal_actions():
      new_state = state.clone()
      new_state.apply_action(action)
      construct_is_mcts_policy(game, new_state, tabular_policy, bot, searched)


def main(_):

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

  game = pyspiel.load_game(FLAGS.game)
  evaluator = pyspiel.RandomRolloutEvaluator(1, SEED)

  for max_simulations in [10, 100, 1000, 10000]:
    print("-------- TRAIN IS MCTS {} Simulations --------".format(max_simulations))
    print("{:>5} {:>10} {:>50} {:>20} {:>20}".format(
      "max_sims", "uct_c", "final_policy_type", "NashConv", "Time"))
    for uct_c in [0.2, 0.5, 1.0, 2.0, 4.0]:
      for final_policy_type in [
          pyspiel.ISMCTSFinalPolicyType.NORMALIZED_VISIT_COUNT,
          pyspiel.ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
          pyspiel.ISMCTSFinalPolicyType.MAX_VALUE
      ]:
        start_time = time.time()
        tabular_policy = policy.TabularPolicy(game)
        bot = pyspiel.ISMCTSBot(SEED, evaluator, uct_c, max_simulations, -1,
                                final_policy_type, False, False)
        searched = {}
        construct_is_mcts_policy(game, game.new_initial_state(), tabular_policy,
                                 bot, searched)
        nash_conv = exploitability.nash_conv(game, tabular_policy)
        time_done = time.time() - start_time
        print("{:>5} {:>10} {:>50} {:>20} {:>25}".format(max_simulations, uct_c,
                                                  str(final_policy_type), nash_conv, time_done))
        # log to Weights & Biases Service
        if final_policy_type == pyspiel.ISMCTSFinalPolicyType.MAX_VALUE:
          if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "UCT C": uct_c,"max. Simulationen": max_simulations, "Zeit": time_done}, commit=True)
    if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "UCT C": uct_c, "Zeit": time_done}, commit=True)



if __name__ == "__main__":
  app.run(main)
