import numpy as np
import matplotlib.pyplot as plt
from main.clustering import metrics
from data.source_datasets.datasets import AbstractDataset
from data.source_datasets.datasets import LBNL_FTP_PKTDataset1
from data.source_datasets.datasets import AllHTTPDatasetsCombined


ae_ftp = "main/clustering/clusterwise_matrix_AE_balanced_ftp.csv"
cnn_ftp = "main/clustering/clusterwise_matrix_CNN_balanced_ftp.csv"
base_ftp = "main/clustering/clusterwise_matrix_Baseline_ftp.csv"

ae_http = "main/clustering/clusterwise_matrix_AE_balanced_http.csv"
cnn_http = "main/clustering/clusterwise_matrix_CNN_balanced_http.csv"
base_http = "main/clustering/clusterwise_matrix_Baseline_http.csv"


def report_clustermetrics(clusmat_path: str):
    clusmat = np.genfromtxt(clusmat_path, delimiter=",")
    acc = metrics.get_accuracy_metric(clusmat)
    acc_dom = metrics.get_accuracy_metric(clusmat, skip_zeros=True)
    conf = metrics.get_confidence_metric(clusmat)
    conf_dom = metrics.get_confidence_metric(clusmat, skip_zeros=True)
    result = {"Path": clusmat_path,
              "Accuracy": acc,
              "Dominant Accuracy": acc_dom,
              "Confidence": conf,
              "Dominant Confidence": conf_dom}
    return result


print(report_clustermetrics(ae_ftp))
print(report_clustermetrics(cnn_ftp))
print(report_clustermetrics(base_ftp))

print(report_clustermetrics(ae_http))
print(report_clustermetrics(cnn_http))
print(report_clustermetrics(base_http))


def get_dataset_balance(dataset: AbstractDataset):
    occurrences = []
    balanced_occurrences = []
    for _ in range(len(dataset.protocol_type.value.keys())):
        occurrences.append(0)
        balanced_occurrences.append(0)
    for data in dataset:
        index = metrics.classify_statement(data, dataset.protocol_type)
        occurrences[index] += 1
    dataset.balance_dataset(class_limit=100)
    for data in dataset:
        index = metrics.classify_statement(data, dataset.protocol_type)
        balanced_occurrences[index] += 1
    return occurrences, balanced_occurrences


ftp_source_dataset = LBNL_FTP_PKTDataset1()
http_source_dataset = AllHTTPDatasetsCombined()
orig_occ_ftp, bal_occ_ftp = get_dataset_balance(ftp_source_dataset)
orig_occ_http, bal_occ_http = get_dataset_balance(http_source_dataset)

n_groups = len(orig_occ_http)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, orig_occ_http, bar_width, alpha=opacity, color='b', label='Original')
rects2 = plt.bar(index + bar_width, bal_occ_http, bar_width, alpha=opacity, color='g', label='Balanced')

plt.xlabel('HTTP Types')
plt.ylabel('Num Occurrences')
plt.title('HTTP Dataset Type Distribution')
plt.legend()

plt.tight_layout()
plt.show()

n_groups = len(orig_occ_ftp)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, orig_occ_ftp, bar_width, alpha=opacity, color='b', label='Original')
rects2 = plt.bar(index + bar_width, bal_occ_ftp, bar_width, alpha=opacity, color='g', label='Balanced')

plt.xlabel('FTP Types')
plt.ylabel('Num Occurrences')
plt.title('FTP Dataset Type Distribution')
plt.legend()

plt.tight_layout()
plt.show()
print("")
