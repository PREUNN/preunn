import numpy as np


def classify_http_statement(http_statement: str):
    """
    Supparize http statements important information
    :param http_statement: http data string
    :return: Type, length and code
    """
    length = http_statement.find("\r\n\r\n")
    fragment_list = http_statement.split("\r\n")
    try:
        http_type, fragment_list[0] = fragment_list[0].split(" ", maxsplit=1)
    except:
        http_type = "Broken"
    number_of_lines = len(fragment_list) - 2
    http_code = "-"
    if http_type == "HTTP/1.1":
        http_code = fragment_list[0].split(" ", maxsplit=1)[0]
    return http_type + "\n" + str(length) + ", " + http_code


def get_clustering_accuracy(clusters: []):
    """
    Evaluation method for self-organizing maps
    :param clusters: List of clusters
    :return: Accuracy found
    """
    check_list = ["GET", "HTTP/1.1", "placeholder1", "placeholder2", "POST",
                  "HEAD", "DELETE", "OPTIONS", "PUT", "TRACE", "CONNECT"]
    accuracy_matrix = np.zeros([16, 11])
    for i in range(len(clusters)):
        try:
            winner = clusters[i][1]
            header, rest = clusters[i][0].split("\n")
            http_code = rest.split(",")[1]
            if header == "HTTP/1.0":
                continue
            if header != "HTTP/1.1":
                accuracy_matrix[winner][check_list.index(header)] += 1
            else:
                http_code = int(http_code)
                if http_code < 300:
                    accuracy_matrix[winner][1] += 1
                if 300 <= http_code < 400:
                    accuracy_matrix[winner][2] += 1
                if http_code >= 400:
                    accuracy_matrix[winner][3] += 1
        except:
            continue
    return accuracy_matrix


def classify_ftp_statement(statement: str):
    # USER / PASS / 331 / 230
    # PORT / TYPE / MODE / 200 / 200 / 200 + 502 + 504
    # PASV / 227
    # QUIT / 221
    # CWD / 250
    # REST / 350
    # LIST / RETR / SIZE / MDTM / 150 / 150 + 550 / 213 + 550 / 213 + 550
    # HELP / SYST / 214 / 215
    # ALLO / 202
    # 220 Service ready for new user.(Welcome - Banner)
    # Sonstiges
    # 226 Closing data connection = > Nach RETR und PORT
    ftp_type_list = {"Type1": ["USER", "PASS", "331", "230"],
                     "Type2": ["PORT", "TYPE", "MODE", "200", "502", "504"],
                     "Type3": ["PASV", "227"],
                     "Type4": ["QUIT", "221"],
                     "Type5": ["CWD", "250"],
                     "Type6": ["REST", "350"],
                     "Type7": ["LIST", "RETR", "SIZE", "MDTM", "150", "550",
                               "213"],
                     "Type8": ["HELP", "SYST", "214", "215"],
                     "Type9": ["ALLO", "202"],
                     "Type10": ["220"],
                     "Type11": ["226"],
                     "Type12": ["misc"]}

    # check if any keyword is in the statement, return type accordingly
    # miscellaneous is default
    found_type = "Type12"   # == misc
    for type in ftp_type_list:
        if any(key_word in statement for key_word in ftp_type_list[type]):
            found_type = type

    return found_type
