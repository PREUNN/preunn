import random
import re
from scapy.all import wrpcap, Ether, IP, TCP
from data.postprocessing.sequence_postprocessing.processing import \
    sequence_tensor_to_string_list
from models.long_short_term_memory.personal_trainer import \
    LongShortTermMemoryPersonalTrainer
from main.clustering.metrics import get_label


def get_new_http_statements(trainer: LongShortTermMemoryPersonalTrainer,
                            num_classes: int):
    """
    use as fuzzer base
    :return:
    """
    splitter = ''
    for x in range(num_classes):
        splitter += 'EOP°' + str(x) + '\n\n|'
        splitter += 'SOP°' + str(x) + '\n|'
    splitter = splitter[:-1]
    statement_list = []
    package_list = []

    # iterating over test data for initialization vectors
    for _, (item, _) in enumerate(trainer.test_data):
        sample_sequence, _ = trainer.sample(random_delimiter=3,
                                            length=item.shape[1],
                                            data=item.to(trainer.device))

        # evaluating each sample separately and create network packages
        for each in sample_sequence:
            each = sequence_tensor_to_string_list(each.unsqueeze(0))[0]
            temp_list = re.split(splitter, each)
            for each in temp_list[1:-1]:
                if each != "":
                    print(each)
                    statement_list.append(each)
                    address = str(random.randint(1, 192)) + "." + \
                              str(random.randint(1, 192)) + "." + \
                              str(random.randint(1, 192)) + "." + \
                              str(random.randint(1, 192))
                    package = Ether() / IP(dst=address) \
                              / TCP(dport=21, flags='S') / each
                    package_list.append(package)
            if len(package_list) > 1000:
                package_list = package_list[:1000]
                break
        wrpcap("test_http.pcap", package_list)
        break
    return


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
