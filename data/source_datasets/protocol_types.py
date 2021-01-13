import enum


class Protocol(enum.Enum):
    """
    Types of protocols
    """
    HTTP = {0: ["##################"],
            1: ["GET "],
            2: ["HTTP/1.1 2"],
            3: ["HTTP/1.1 3"],
            4: ["HTTP/1.1 4"],
            5: ["POST "],
            6: ["HEAD "],
            7: ["DELETE "],
            8: ["OPTIONS "],
            9: ["PUT "],
            10: ["TRACE "],
            11: ["CONNECT "]}
    FTP = {0: ["##################"],
           1: ["ACCT", "ADAT", "AUTH", "CONF", "ENC", "MIC",
               "PASS", "PBSZ", "PROT", "QUIT", "USER"],
           2: ["230", "331", "332", "530", "532"],
           3: ["PASV", "EPSV", "LPSV"],
           4: ["227", "228", "229"],
           5: ["ABOR", "EPRT", "LPRT", "MODE", "PORT", "REST",
               "RETR", "TYPE", "XSEM", "XSEN"],
           6: ["125", "150", "221", "225", "226", "421", "425",
               "426"],
           7: ["ALLO", "APPE", "CDUP", "CWD", "DELE", "LIST",
               "MKD", "MDTM", "PWD", "RMD", "RNFR", "RNTO",
               "STOR", "STRU", "SYST", "XCUP", "XMKD", "XPWD",
               "XRMD"],
           8: ["212", "213", "215", "250", "257", "350", "532"],
           9: ["120", "200", "202", "211", "214", "220", "450",
               "451", "452", "500", "501", "502", "503", "504",
               "550", "551", "552", "553", "554", "555"]}
    # DNS = 3       # not yet available
