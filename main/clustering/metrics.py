import copy
import numpy as np
from models.self_organizing_map.personal_trainer import SelfOrganizingMapPersonalTrainer
from data.source_datasets.protocol_types import Protocol
from data.postprocessing.image_postprocessing.processing import image_tensor_to_string_list


def get_accuracy_matrix(trainer: SelfOrganizingMapPersonalTrainer):
    """
    Calculate and return the accuracy matrix for a given som and test data.

    :param trainer: Trained MiniSom model pt to be evaluated.
    :return: Numpy array of num clusters * num classes from the som.
    """

    # constants
    protocol = trainer.test_data.dataset.source_dataset.protocol_type
    num_classes = trainer.model.get_weights().shape[1]
    num_clusters = max(len(Protocol.HTTP.value.keys()), len(Protocol.FTP.value.keys()))

    winner_list = []
    for _, (item, _) in enumerate(trainer.test_data):
        new_data = trainer.get_new_data(item)
        winners = np.array([trainer.model.winner(x)[1] for x in new_data.detach()])

        data_strings = image_tensor_to_string_list(item)
        winner_list.extend([(classify_statement(item, protocol), win) for item, win in zip(data_strings, winners)])

    accuracy_matrix = np.zeros([num_classes, num_clusters])
    for type, winner in winner_list:
        accuracy_matrix[winner][type] += 1

    # rounding to 3 digits
    accuracy_matrix = 1000 * accuracy_matrix
    accuracy_matrix = np.rint(accuracy_matrix)
    accuracy_matrix /= 1000
    return accuracy_matrix


def get_clusterwise_share(accuracy_matrix: np.ndarray) -> np.ndarray:
    """
    Get clusterwise share version of the accuracy matrix.
    :param accuracy_matrix: original accuracy matrix
    :return: clusterwise share matrix
        """
    clusterwise_matrix = copy.deepcopy(accuracy_matrix)
    # get clusterwise share
    for i in range(len(clusterwise_matrix)):
        clusterwise_matrix[i] /= max(sum(clusterwise_matrix[i]), 1)
    clusterwise_matrix = 1000 * clusterwise_matrix
    clusterwise_matrix = np.rint(clusterwise_matrix)
    clusterwise_matrix /= 10
    return clusterwise_matrix


def get_typewise_share(accuracy_matrix: np.ndarray) -> np.ndarray:
    """
    Get typewise share version of the accuracy matrix.
    :param accuracy_matrix: original accuracy matrix
    :return: typewise share matrix
    """
    typewise_matrix = copy.deepcopy(accuracy_matrix)
    # get typewise share
    for j in range(len(typewise_matrix[0])):
        col_max = 0
        for i in range(len(typewise_matrix)):
            col_max += typewise_matrix[i][j]
        for i in range(len(typewise_matrix)):
            typewise_matrix[i][j] /= max(col_max, 1)
    typewise_matrix = 1000 * typewise_matrix
    typewise_matrix = np.rint(typewise_matrix)
    typewise_matrix /= 10
    return typewise_matrix


def get_accuracy_metric(clusterwise_matrix: np.ndarray, skip_zeros: bool = False) -> float:
    """
    This metric counts the number of confident clusters with more than 50%
    confidence.
    :param clusterwise_matrix: original clusterwise matrix
    :param skip_zeros: whether or not to count clusters with no elements.
    :return: relative number of confident clusters
    """
    num_conf_cluster = 0
    num_clusters = clusterwise_matrix.shape[0]
    for row in clusterwise_matrix:
        if skip_zeros and sum(row) == 0.0:
            num_clusters -= 1
            continue
        for cell in row:
            if cell > 50.00:
                num_conf_cluster += 1
    rel_conf_cluster = num_conf_cluster / num_clusters
    return rel_conf_cluster


def get_confidence_metric(clusterwise_matrix: np.ndarray, skip_zeros: bool = False) -> float:
    """
    This metric calculates the average confidence of the clusters.
    :param clusterwise_matrix: original clusterwise matrix
    :param skip_zeros: whether or not to count clusters with no elements.
    :return: average confidence
    """
    num_conf_cluster = 0
    ttl_confidence = 0.0
    num_clusters = clusterwise_matrix.shape[0]
    for row in clusterwise_matrix:
        if skip_zeros and sum(row) == 0.0:
            num_clusters -= 1
            continue
        for cell in row:
            if cell > 50.00:
                num_conf_cluster += 1
                ttl_confidence += cell
    avg_confidence = ttl_confidence / num_clusters
    return avg_confidence


def get_label(cluster: int, accuracy_matrix: np.ndarray) -> int:
    """
    This method will return the label index for a given cluster from the
    accuracy matrix.
    :param cluster: index of a cluster.
    :param accuracy_matrix: original accuracy matrix.
    :return: index of the class, -1, if no class found
    """
    row = accuracy_matrix[cluster]
    max = row[0]
    index = -1
    for i, cell in enumerate(row):
        if cell > max:
            max = cell
            index = i
    return index


def classify_statement(statement: str, protocol: Protocol):
    """
    Find the type of a given http statement
    :param statement: data string
    :param protocol: protocol type of the statement
    :return: type of http
    """

    # decide on the type list used
    type_list = protocol.value

    # check if any keyword is in the statement, return type accordingly
    # miscellaneous is default
    found_type = 0  # == misc
    for type in type_list:
        if any(key_word in statement for key_word in type_list[type]):
            found_type = type
    return found_type
