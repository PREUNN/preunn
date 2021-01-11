from main.clustering.metrics import get_label


def get_cluster_prediction_clusterwise(trainer, accuracy_matrix):
    """
    Clusterwise statemachine prediction evaluation metric.
    :return: None
    """
    batch_size = 0
    num_correct = 0
    for batch_id, (item, target) in enumerate(trainer.test_data):
        batch_size = item.shape[0]
        sample = trainer.sample_states(length=1, data=item.to(trainer.device))

        predictions = sample[:, -1].cpu().numpy()
        targets = target[:, -1].cpu().numpy()
        for pred, targ in zip(predictions, targets):
            pred_label = get_label(pred, accuracy_matrix)
            targ_label = get_label(targ, accuracy_matrix)
            if pred_label == targ_label:
                num_correct += 1

    avg_correct = num_correct / (len(trainer.test_data) * batch_size)
    return avg_correct
