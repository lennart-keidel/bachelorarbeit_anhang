""" Helper methods for handeling paths, saving models, and automating training """

import dill
import tensorflow.compat.v1 as tf
import keras
import os


# create path of checkpoint dir, to save
def create_checkpoint_path_from_arguments(name_game, name_model_or_solver, checkpoint_dir="/tmp", is_nn_model=True):
  return "{}/trained_{}_{}_{}".format(checkpoint_dir, name_model_or_solver, name_game, "model" if is_nn_model else "solver.dill")


# save solver model
def save_solver_model(solver_object, path_checkpoint="/tmp", enable_saving=True, name_game="leduc_poker"):
  if enable_saving:
    new_path_checkpoint = create_checkpoint_path_from_arguments(name_game, solver_object.__class__.__name__, path_checkpoint, is_nn_model=False)
    print("SAVING SOLVER UNDER: {}".format(new_path_checkpoint))
    with open(new_path_checkpoint, "wb") as file:
      dill.dump(solver_object, file, dill.HIGHEST_PROTOCOL)


# load solver model
def load_solver_model(solver_name, path_checkpoint="/tmp", enable_load=True, name_game="leduc_poker"):
  if enable_load:
    new_path_checkpoint = create_checkpoint_path_from_arguments(name_game, solver_name, path_checkpoint, is_nn_model=False)
    print("LOADING SOLVER FROM: {}".format(new_path_checkpoint))
    if os.path.exists(new_path_checkpoint):
      with open(new_path_checkpoint, "rb") as file:
        return dill.load(file)
    else:
      return False


# save NN models
def save_models(array_agents, path_checkpoint="/tmp", enable_saving=True, name_game="leduc_poker"):
  if enable_saving:
    for f in range(len(array_agents)):
      class_name=array_agents[f].__class__.__name__
      new_path_checkpoint = create_checkpoint_path_from_arguments(name_game, class_name, path_checkpoint, is_nn_model=True)
      if class_name=="DeepCFRSolver":
        array_agents[f].save_policy_network(new_path_checkpoint)
      else:
        array_agents[f].save(new_path_checkpoint)



# reload NN models
def load_models(tf_session=False, array_agents=[], path_checkpoint="/tmp", enable_load=True, name_game="leduc_poker"):
  if enable_load:
    for f in range(len(array_agents)):
      class_name=array_agents[f].__class__.__name__
      new_path_checkpoint = create_checkpoint_path_from_arguments(name_game, class_name, path_checkpoint, is_nn_model=True)
      if(array_agents[f].has_checkpoint(new_path_checkpoint)):
        if class_name=="DeepCFRSolver":
          array_agents[f] = keras.models.load_model(new_path_checkpoint)
        else:
          array_agents[f].restore(new_path_checkpoint)
  else:
    tf_session.run(tf.global_variables_initializer())



def is_nash_convergence_good_enough(nash_conv_now, nash_conv_before, time_training_in_seconds, end_after_n_hours=3):
  if abs(nash_conv_before - nash_conv_now) < 0.0001 and nash_conv_now < 0.0001:
    return True
  if time_training_in_seconds > (end_after_n_hours*60*60):
    return True
  return False
