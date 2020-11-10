from matplotlib import pyplot as plt

from data.source_datasets.datasets import AllHTTPDatasetsCombined, \
    LBNL_FTP_PKTDatasetCombined

http_dataset = AllHTTPDatasetsCombined()
ftp_dataset = LBNL_FTP_PKTDatasetCombined()

http_types = ["OPTIONS", "GET", "HEAD", "POST", "PUT", "DELETE", "TRACE",
              "CONNECT", "HTTP/1.1"]


def plot_length_sorted(dataset, color="blue"):
    plt.plot(sorted([len(entry) for entry in http_dataset]), color=color)
    plt.xlabel("Different " + str(dataset.protocol_type) + " messages")
    plt.ylabel("Length of message")
    plt.show()


def plot_length_occ(dataset, color="blue"):
    dic = {i: 0 for i in range(max(len(entry) for entry in dataset) + 1)}
    for each in dataset:
        dic[len(each)] += 1
    plt.plot(list(dic.keys()), list(dic.values()), color=color)
    plt.xlabel("Length of message")
    plt.ylabel("Number of " + str(dataset.protocol_type) + " messages")
    plt.show()


def find_all_scii_zeros():
    count = 0
    for each in http_dataset:
        if each.find(str(b'\x00')) > 0:
            print(each)
            count += 1
    print(count)


def show_clusters(dataset, clusters, color="blue"):

    dic = {each: 0 for each in http_types}
    for entry in dataset:
        for key in dic.keys():
            if key in entry:
                dic[key] += 1
    plt.bar(list(dic.keys()), list(dic.values()), color=color)
    plt.xlabel("Type of HTTP message")
    plt.ylabel("Number of HTTP messages")
    plt.show()


