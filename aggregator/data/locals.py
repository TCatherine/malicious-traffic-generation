from pathlib import Path

dataset_folder = Path(__file__).parent / 'dataset'

xss = {
    'path': dataset_folder / 'xss',
    're': r"GET ([^\n]*)[\n\s]*"
          r"HTTP\/([^\n]*)\n"
          r"Host: ([^\n]*)\n"
          r"User-Agent: ([^\n]*)\n"
          r"Accept-Encoding: ([^\n]*)\n"
          r"Accept: ([^\n]*)\n"
          r"Connection: ([^\n]*)\n"
}

benign = {
    'path': dataset_folder / 'benign',
    're': r"GET ([^\s]*) HTTP\/([^\n]*)\n"
          r"HOST\s*: ([^\n]*)\n"
          r"USER-AGENT\s*: ([^\n]*)\n"
          r"ACCEPT\s*: ([^\n]*)\n"
          r"PROXY-CONNECTION\s*: ([^\n]*)"
}

xss_url = {
    'path': dataset_folder / 'xss_url',
    're': r"([^\n]*)\n"
}

xss_url_suricata = {
    'path': dataset_folder / 'xss_url_suricata',
    're': r"([^\n]*)\n"
}
