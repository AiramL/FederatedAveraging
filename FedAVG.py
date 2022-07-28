#
# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformática e Automação (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#

import tensorflow as tf

# receives a list of clients' models, and the size of each client partition, returns the new global model
def federated_average(model_list: list, data_size_list: list):

    # check if the list is empty
    if not model_list:
        return None

    total_data = sum(data_size_list)

    # averaging the models in the list
    for model_index, model in enumerate(model_list):

        for weight_index in range(len(model.weights)):

            if not model_index:
                # get the first model layers to do the average
                global_model_update = []
                for index in range(len(model_list[0].weights)):
                    global_model_update.append(model_list[0].weights[index]*data_size_list[model_index]/total_data)
            else:
                # update the weights using the remaning models
                global_model_update[weight_index] += model.weights[weight_index]*data_size_list[model_index]/total_data


    # get the model structure
    new_model = tf.keras.models.clone_model(model_list[0])
    
    # update model weights
    new_model.set_weights(global_model_update)


    return new_model
