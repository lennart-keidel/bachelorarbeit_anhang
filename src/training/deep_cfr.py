""" train Deep CFR, Deep Counterfactual Regret Minimization """

from absl import app
from absl import flags
from absl import logging
import os
import time

import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score, exploitability, deep_cfr
import pyspiel
from src.methods import *
from src.shared_flags import *

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_enum("training_agent", "CFRBRSolver", VALID_PLAYERS, "Name of Agent to train")
flags.DEFINE_integer("num_iterations", 400, "Number of iterations")
flags.DEFINE_integer("num_traversals", 10, "Number of traversals/games")


def has_checkpoint(tf_session, checkpoint_dir):
  for name, _ in self._savers:
    if tf.train.latest_checkpoint(
        self._full_checkpoint_name(checkpoint_dir, name),
        os.path.join(checkpoint_dir,
                      self._latest_checkpoint_filename(name))) is None:
      return False
  return True


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
  game = pyspiel.load_game(FLAGS.game)
  with tf.Session() as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=(16,),
        advantage_network_layers=(16,),
        num_iterations=FLAGS.num_iterations,
        num_traversals=FLAGS.num_traversals,
        learning_rate=1e-3,
        batch_size_advantage=128,
        batch_size_strategy=1024,
        memory_capacity=1e7,
        policy_network_train_steps=400,
        advantage_network_train_steps=20,
        reinitialize_advantage_networks=False)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # create path for saving checkpoint
    # create seperate directory for this model
    new_path_checkpoint = create_checkpoint_path_from_arguments(FLAGS.game, deep_cfr_solver.__class__.__name__, FLAGS.save_checkpoint_dir, is_nn_model=True)
    if not os.path.exists(new_path_checkpoint):
      os.makedirs(new_path_checkpoint)
    new_path_checkpoint += "/model"

    # train agent
    start_time = time.time()
    nash_conv=-1
    nash_conv_before=-100
    for ep in range(FLAGS.training_number_episodes):
      print("-------- TRAIN DeepCFR ITERATION {} --------".format(ep))
      _, advantage_losses, policy_loss = deep_cfr_solver.solve()
      time_done = time.time() - start_time

      # save model
      print("SAVING MODEL ...")
      saver.save(sess, new_path_checkpoint)

      # to restore the model
      # saver.restore(sess, new_path_checkpoint)

      # logging
      for player, losses in advantage_losses.items():
        print("Advantage for player %d: %s", player,
                    losses[:2] + ["..."] + losses[-2:])
        print("Advantage Buffer Size for player %s: '%s'", player,
                    len(deep_cfr_solver.advantage_buffers[player]))
      print("Strategy Buffer Size: '%s'",
                  len(deep_cfr_solver.strategy_buffer))
      print("Final policy loss: '%s'", policy_loss)
      print("Time so far: {}".format(time_done))


      average_policy = policy.tabular_policy_from_callable(
          game, deep_cfr_solver.action_probabilities)


      nash_conv = exploitability.nash_conv(game, average_policy)
      # log to Weights & Biases Service
      if FLAGS.wandb_enable: wandb.log({ "Nash Convergenz": nash_conv, "Iteration": ep, "Zeit": time_done}, commit=True)
      print("Deep CFR in '%s' - NashConv: %s", FLAGS.game, nash_conv)

      average_policy_values = expected_game_score.policy_value(
          game.new_initial_state(), [average_policy] * 2)
      print("Computed player 0 value: {}".format(average_policy_values[0]))
      print("Expected player 0 value: {}".format(-1 / 18))
      print("Computed player 1 value: {}".format(average_policy_values[1]))
      print("Expected player 1 value: {}".format(1 / 18))

      if is_nash_convergence_good_enough(nash_conv_now=nash_conv, nash_conv_before=nash_conv_before , time_training_in_seconds=(time.time() - start_time), end_after_n_hours=6):
        exit()
      nash_conv_before=nash_conv


if __name__ == "__main__":
  app.run(main)
