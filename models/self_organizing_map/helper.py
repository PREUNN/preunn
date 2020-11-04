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
    check_list = ["GET", "HTTP/1.1", "placeholder1", "placeholder2", "POST", "HEAD", "DELETE", "OPTIONS", "PUT", "TRACE", "CONNECT"]
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

