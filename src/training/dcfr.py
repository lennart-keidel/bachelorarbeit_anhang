""" train DCFR, Discounted Counterfactual Regret Minimization """

from absl import app, flags

from absl import app
from absl import flags

import pyspiel
from open_spiel.python.algorithms import discounted_cfr, exploitability
import time


from src.shared_flags import *
from src.methods import *


flags.DEFINE_enum("training_agent", "DCFRSolver", VALID_PLAYERS, "Name of Agent to train")

FLAGS = flags.FLAGS

def main(_):

  print("-------- TRAIN {} --------".format(FLAGS.training_agent))

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

  game = pyspiel.load_game(
        FLAGS.game,
        {"players": FLAGS.game_number_players}
    )

  solver = discounted_cfr.DCFRSolver(game)

  # load trained solver
  loaded_solver = load_solver_model(solver_name=solver.__class__.__name__, path_checkpoint=FLAGS.save_checkpoint_dir, enable_load=FLAGS.load_enable ,name_game=FLAGS.game)
  if loaded_solver: solver = loaded_solver

  start_time = time.time()
  nash_conv=-1
  nash_conv_before=-100
  for ep in range(int(FLAGS.training_number_episodes)):
    if FLAGS.training_agent == "XFPSolver":
      solver.iteration()
    else:
      solver.evaluate_and_update_policy()

    # log
    time_done = time.time() - start_time
    nash_conv = exploitability.nash_conv(game, solver.average_policy())
    print("Iteration {} Nash Convergence: {:.6f}".format(
        ep, nash_conv))
    print("Time so far: {}".format(time_done))

    if (ep+1) % FLAGS.save_frequency == 0:

      # log to Weights & Biases Service
      if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "Iteration": ep, "Zeit": time_done}, commit=True)

      print("SAVING MODEL ...")
      save_solver_model(solver_object=solver, path_checkpoint=FLAGS.save_checkpoint_dir, enable_saving=FLAGS.save_enable ,name_game=FLAGS.game)
      nash_conv_before=nash_conv
    else:

      # log to Weights & Biases Service
      if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "Iteration": ep, "Zeit": time_done}, commit=False)

    # exit training if good enough
    if is_nash_convergence_good_enough(nash_conv_now=nash_conv, nash_conv_before=nash_conv_before , time_training_in_seconds=(time.time() - start_time), end_after_n_hours=2):
      # log to Weights & Biases Service
      if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "Iteration": ep, "Zeit": time_done}, commit=True)
      save_solver_model(solver_object=solver, path_checkpoint=FLAGS.save_checkpoint_dir, enable_saving=FLAGS.save_enable ,name_game=FLAGS.game)
      exit()



if __name__ == "__main__":
  app.run(main)
