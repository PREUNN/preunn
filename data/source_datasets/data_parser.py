from data.source_datasets.protocol_types import Protocol
import scapy.all
import abc
import os


class AbstractProtocolParser(abc.ABC):
    """Abstract parser class for each protocol"""

    def __init__(self, protocol_type: Protocol):
        self.protocol_type = protocol_type

    def get_protocol_type(self):
        """
        Getter for the type of protocol used
        :return: Type of protocol of this parser
        """
        return self.protocol_type

    @abc.abstractmethod
    def parse(self, filepath):
        pass


class HTTPParser(AbstractProtocolParser):
    """Parser class for HTTP datasets"""

    def __init__(self):
        super().__init__(Protocol.HTTP)

    def parse(self, filepath):
        """
        This is the main reading method for the provided http datasets as given
        in the path variables above. The method reads pcap files at location
        filepath. Also filters bad http statements
        :param filepath: Location of the dataset pcap
        :return: List of all the read http statements
        """
        # parameter checks
        assert os.path.isfile(filepath)

        sessions = scapy.all.rdpcap(filepath).sessions()
        i = 0
        return_dataset = []

        # iterating over sessions
        for session in sessions:
            http_payload = ""

            # iterating over packages
            for packet in sessions[session]:
                try:
                    packet = packet.payload.payload

                    # get application layer data
                    if packet.dport == 80 or packet.sport == 80:
                        data = packet.payload.original

                        # End of http header, only data follows
                        end_index = data.find(b'\x0d\x0a\x0d\x0a') + 4
                        http_payload += str(data[:end_index].decode("utf-8"))
                        break
                except:
                    break

            # check for validity of package
            check_validity_list = ["OPTIONS", "GET", "HEAD", "POST", "PUT",
                                   "DELETE", "TRACE", "CONNECT", "HTTP/1.1"]
            if any(method in http_payload for method in check_validity_list) \
                    and len(http_payload) > 20:
                if any(ord(c) > 127 for c in http_payload):
                    print(http_payload)
                    continue
                return_dataset.append(http_payload)
                i += 1

        # output
        return return_dataset


class FTPParser(AbstractProtocolParser):
    """Parser class for FTP datasets"""

    def __init__(self):
        super().__init__(Protocol.FTP)

    def parse(self, filepath):

        packets = scapy.all.rdpcap(open(filepath, 'rb'))
        return_dataset = []

        # iterate through every packet
        for packet in packets:
            try:
                packet_payload = packet.payload.payload

                if packet.dport == 21 or packet.sport == 21:
                    data = packet_payload.payload.original
                    return_dataset.append(str(data.decode("utf-8")))
            except:
                continue

        # Output
        return return_dataset
