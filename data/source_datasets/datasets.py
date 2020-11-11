from data.source_datasets.data_parser import AbstractProtocolParser, \
    HTTPParser, FTPParser
from torch.utils.data import Dataset
from abc import ABC
import random
import torch
import copy
import os


class AbstractDataset(Dataset, ABC):
    """Abstract base class for datasets based on a file path"""

    def __init__(self, filepath: str, protocol_parser: AbstractProtocolParser):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.filepath = filepath
        self.parser = protocol_parser
        self.protocol_type = self.parser.get_protocol_type()
        self.data_min_value = 0
        self.data_max_value = 127
        self.data = self.retrieve_dataset()
        self.length = len(self.data)

    def retrieve_dataset(self):
        """
        Loading routine for the dataset, will reload and save if the parsed data
        is unavailable
        :return: Loaded/parsed dataset
        """
        # file already exists parsed
        if os.path.isfile(self.dir_path + self.filepath):
            ret = torch.load(self.dir_path + self.filepath)
            print("loaded dataset " + self.filepath)

        # file does not already exist parsed
        else:
            ret = self.parser.parse(self.dir_path + self.filepath + ".pcap")
            torch.save(ret, self.dir_path + self.filepath)
            print("reloaded and saved dataset " + self.filepath)
        return ret

    def __len__(self):
        self.length = len(self.data)
        return self.length

    def __getitem__(self, idx):
        assert idx <= self.__len__()
        return self.data[idx]

    def merge(self, new_dataset: Dataset):
        """
        Merging a new dataset with this one
        :param new_dataset: New dataset to be merged into this
        :return: None
        """
        # parameter checks
        assert self.protocol_type == new_dataset.protocol_type

        self.data.extend(new_dataset.data)

    def split(self, split_value: float):
        """
        Function to split the dataset and returns 2 new datasets accordingly
        :param split_value: Value of how to split the data. Needs to be inside
        the interval [0, 1]
        :return: 2 new datasets
        """
        # parameter checks
        assert 0 <= split_value <= 1

        # splitting data
        split_index = int(self.__len__() * split_value)
        first_dataset = copy.deepcopy(self)
        second_dataset = copy.deepcopy(self)

        # output
        first_dataset.data = self.data[:split_index]
        second_dataset.data = self.data[split_index:]
        return first_dataset, second_dataset

    def shuffle_dataset(self):
        """
        Randomly shuffle data inside the dataset. Seed is set for repeatablility
        :return: None
        """
        random.Random(123456789).shuffle(self.data)


class AbstractHTTPDataset(AbstractDataset):
    """Abstract class for all datasets of protocol type HTTP"""

    def __init__(self, filepath: str):
        super().__init__(filepath, HTTPParser())


class AbstractFTPDataset(AbstractDataset):
    """Abstract class for all datasets of protocol type FTP"""

    def __init__(self, filepath: str):
        super().__init__(filepath, FTPParser())


class IEEEHTTPDataset1(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/IEEEHTTPandDNS/HTTP/http_set1_1")


class IEEEHTTPDataset2(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/IEEEHTTPandDNS/HTTP/http_set1_2")


class IEEEHTTPDataset3(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/IEEEHTTPandDNS/HTTP/http_set1_3")


class NitrobaHTTPDataset(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/Nitroba/nitroba_http")


class CICDDoS2019HTTPDataset(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/CICDDoS2019/DDOSsmall")


class ISCX_IDS_2012Dataset1(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP1")


class ISCX_IDS_2012Dataset2(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP2")


class ISCX_IDS_2012Dataset3(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP3")


class ISCX_IDS_2012Dataset4(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP4")


class ISCX_IDS_2012Dataset5(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP5")


class ISCX_IDS_2012Dataset6(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP6")


class ISCX_IDS_2012Dataset7(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP7")


class ISCX_IDS_2012Dataset8(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP8")


class ISCX_IDS_2012Dataset9(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP9")


class ISCX_IDS_2012Dataset10(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP10")


class ISCX_IDS_2012Dataset11(AbstractHTTPDataset):
    def __init__(self):
        super().__init__("/HTTP/ISCX-IDS-2012/IDS-HTTP11")


class AllHTTPDatasetsCombined(AbstractHTTPDataset):
    """
    This dataset loads and combines all available HTTP datasets at once
    """

    def __init__(self):
        super().__init__("/HTTP/IEEEHTTPandDNS/HTTP/http_set1_1")
        datasets = []
        datasets.append(IEEEHTTPDataset2())
        datasets.append(IEEEHTTPDataset3())
        datasets.append(NitrobaHTTPDataset())
        datasets.append(CICDDoS2019HTTPDataset())
        datasets.append(ISCX_IDS_2012Dataset1())
        datasets.append(ISCX_IDS_2012Dataset2())
        datasets.append(ISCX_IDS_2012Dataset3())
        datasets.append(ISCX_IDS_2012Dataset4())
        datasets.append(ISCX_IDS_2012Dataset5())
        datasets.append(ISCX_IDS_2012Dataset6())
        datasets.append(ISCX_IDS_2012Dataset7())
        datasets.append(ISCX_IDS_2012Dataset8())
        datasets.append(ISCX_IDS_2012Dataset9())
        datasets.append(ISCX_IDS_2012Dataset10())
        datasets.append(ISCX_IDS_2012Dataset11())
        for dataset in datasets:
            self.merge(dataset)


class LBNL_FTP_PKTDataset1(AbstractFTPDataset):
    def __init__(self):
        super().__init__("/FTP/LBNL-FTP-PKT/lbnl-ftp-only_part1")


class LBNL_FTP_PKTDataset2(AbstractFTPDataset):
    def __init__(self):
        super().__init__("/FTP/LBNL-FTP-PKT/lbnl-ftp-only_part2")


class LBNL_FTP_PKTDataset3(AbstractFTPDataset):
    def __init__(self):
        super().__init__("/FTP/LBNL-FTP-PKT/lbnl-ftp-only_part3")


class LBNL_FTP_PKTDataset4(AbstractFTPDataset):
    def __init__(self):
        super().__init__("/FTP/LBNL-FTP-PKT/lbnl-ftp-only_part4")


class LBNL_FTP_PKTDatasetCombined(AbstractFTPDataset):
    """
    This dataset loads and combines all available FTP datasets at once
    """

    def __init__(self):
        super().__init__("/FTP/LBNL-FTP-PKT/lbnl-ftp-only_part1")
        datasets = []
        datasets.append(LBNL_FTP_PKTDataset2())
        datasets.append(LBNL_FTP_PKTDataset3())
        datasets.append(LBNL_FTP_PKTDataset4())
        for dataset in datasets:
            self.merge(dataset)


from matplotlib import pyplot as plt

if __name__ == "__main__":
    ftp = LBNL_FTP_PKTDatasetCombined()
    count = 0
    for x in ftp:
        try:
            int(x[:2])
        except:
            count += 1
    print(count)
    # data = AllHTTPDatasetsCombined()
    # check_validity_list = ["OPTIONS", "GET", "HEAD", "POST", "PUT",
    #                        "DELETE", "TRACE", "CONNECT", "HTTP/1.1"]
    # dic = {i: 0 for i in check_validity_list}
    # for each in data:
    #     for key in check_validity_list:
    #         if key in each:
    #             dic[key] += 1
    # print(dic)
    # plt.bar(dic.keys(), dic.values())
    # plt.xlabel("HTTP Type")
    # plt.ylabel("Number of instances")
    # plt.show()
    # print("")