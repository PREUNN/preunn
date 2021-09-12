import pandas as pd
from data.source_datasets.datasets import AllHTTPDatasetsCombined, CustomHTTPDataset
from data.source_datasets.datasets import LBNL_FTP_PKTDataset1, CustomFTPDataset
from data.source_datasets.protocol_types import Protocol
from main.clustering.metrics import classify_statement
import matplotlib.pyplot as plt


# all the source datasets
http_source_dataset = AllHTTPDatasetsCombined()
ftp_source_dataset = LBNL_FTP_PKTDataset1()
# source_dataset, _ = source_dataset.split(0.01)
# source_dataset.shuffle_dataset()
# source_dataset.balance_dataset(class_limit=100)


def get_length_ratio(dataset, length_limit=1024):
    df = pd.DataFrame(dataset.data)
    mask = df.iloc[:, 0].str.len() > 1024
    return len(df.loc[mask])/len(df)


def plot_length_occ(dic, protocol, color="blue"):
    plt.bar(list(dic.keys()), list(dic.values()), color=color)
    plt.xlabel("Length of message")
    plt.ylabel("Number of " + str(protocol) + " messages")



print(f"http dataset lengths over 1024 ratio: {get_length_ratio(http_source_dataset)}")
print(f"ftp dataset lengths over 1024 ratio: {get_length_ratio(ftp_source_dataset)}")


fre_http_results = CustomHTTPDataset(filepath="/HTTP/fre_test_http")
sg_http_results = CustomHTTPDataset(filepath="/HTTP/sg_test_http")
fre_ftp_results = CustomFTPDataset(filepath="/FTP/fre_test_ftp")
sg_ftp_results = CustomFTPDataset(filepath="/FTP/sg_test_ftp")
fre_http_stats = {}
sg_http_stats = {}
fre_ftp_stats = {}
sg_ftp_stats = {}
for type in Protocol.HTTP.value.keys():
    fre_http_stats[type] = 0
    sg_http_stats[type] = 0
for statement in fre_http_results:
    fre_http_stats[classify_statement(statement, Protocol.HTTP)] += 1
for statement in sg_http_results:
    sg_http_stats[classify_statement(statement, Protocol.HTTP)] += 1

for type in Protocol.FTP.value.keys():
    fre_ftp_stats[type] = 0
    sg_ftp_stats[type] = 0
for statement in fre_ftp_results:
    fre_ftp_stats[classify_statement(statement, Protocol.FTP)] += 1
for statement in sg_ftp_results:
    sg_ftp_stats[classify_statement(statement, Protocol.FTP)] += 1
for type in fre_ftp_stats:
    fre_ftp_stats[type] /= len(fre_ftp_results)
    sg_ftp_stats[type] /= len(sg_ftp_results)


# plot_length_occ(fre_http_stats, Protocol.HTTP)
# plot_length_occ(sg_http_stats, Protocol.HTTP, color="orange")

df_http = pd.DataFrame({'feature reverse engineering': fre_http_stats,
                        'sequence generation': sg_http_stats})
df_ftp = pd.DataFrame({'feature reverse engineering': fre_ftp_stats,
                       'sequence generation': sg_ftp_stats})

ax = df_http.plot.bar(rot=0)
ax = df_ftp.plot.bar(rot=0)
plt.show()
print("")


