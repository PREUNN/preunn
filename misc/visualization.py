from matplotlib import pyplot as plt

from data.source_datasets.datasets import AllHTTPDatasetsCombined, \
    LBNL_FTP_PKTDatasetCombined

http_dataset = AllHTTPDatasetsCombined()
ftp_dataset = LBNL_FTP_PKTDatasetCombined()

http_types = ["OPTIONS", "GET", "HEAD", "POST", "PUT", "DELETE", "TRACE",
              "CONNECT", "HTTP/1.1 2", "HTTP/1.1 3", "HTTP/1.1 4", "HTTP/1.1 5"]
# http_types = ["OPTIONS", "GET", "HEAD", "POST", "PUT", "DELETE", "TRACE",
#               "CONNECT", "HTTP/1.1"]

ftp_commands = ["ABOR", "ACCT", "ADAT", "ALLO", "APPE", "AUTH", "CCC", "CDUP",
                "CONF", "CWD", "DELE", "ENC", "EPRT", "EPSV", "FEAT", "HELP",
                "LANG", "LIST", "LPRT", "LPSV", "MDTM", "MIC", "MKD", "MLSD",
                "MLST", "MODE", "NLST", "NOOP", "OPTS", "PASS", "PASV", "PBSZ",
                "PORT", "PROT", "PWD", "QUIT", "REIN", "REST", "RETR", "RMD",
                "RNFR", "RNTO", "SITE", "SIZE", "SMNT", "STAT", "STOR", "STOU",
                "STRU", "SYST", "TYPE", "USER", "XCUP", "XMKD", "XPWD", "XRCP",
                "XRMD", "XRSQ", "XSEM", "XSEN"]
ftp_replys = ["1yz", "2yz", "3yz", "4yz", "5yz", "x0z", "x1z", "x2z", "x3z",
              "x4z", "x5z"]
ftp_codes = [110, 120, 125, 150, 200, 202, 211, 212, 213, 214, 215, 220, 221,
             225, 226, 227, 228, 229, 230, 250, 257, 331, 332, 350, 421, 425,
             426, 450, 452, 500, 501, 502, 503, 504, 521, 522, 530, 532, 550,
             551, 552, 553, 554, 555]
ftp_codes_str = [str(i) for i in ftp_codes]
all_ftp_types = []
all_ftp_types.extend(ftp_codes_str)
all_ftp_types.extend(ftp_commands)
all_ftp_types.extend(ftp_replys)


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


def find_all_ascii_zeros():
    count = 0
    for each in http_dataset:
        if each.find(str(b'\x00')) > 0:
            print(each)
            count += 1
    print(count)


def show_clusters(dataset, clusters, color="blue"):
    dic = {each: 0 for each in clusters}
    count = 0
    for entry in dataset:
        for key in dic.keys():
            if key in entry:
                dic[key] += 1
                count += 1
    plt.bar(list(dic.keys()), list(dic.values()), color=color)
    plt.xticks(range(len(dic.keys())), [])
    plt.xlabel("Type of " + str(dataset.protocol_type) + " message")
    plt.ylabel("Number of " + str(dataset.protocol_type) + " messages")
    plt.show()


# Execution zone
if __name__ == "__main__":
    pass
