from pathlib import Path

dataset_folder = Path(__file__).parent.parent / 'data'

xss = {
    'path': dataset_folder / 'xss',
    're': r"(GET) ([^\n]*)[\n\s]*"
          r"HTTP\/([^\n]*)\n"
          r"Host: ([^\n]*)\n"
          r"User-Agent: ([^\n]*)\n"
          r"Accept-Encoding: ([^\n]*)\n"
          r"Accept: ([^\n]*)\n"
          r"Connection: ([^\n]*)\n"
}

benign = {
    'path': dataset_folder / 'benign',
    're': r"(GET) ([^\s]*) HTTP\/([^\n]*)\n"
          r"HOST\s*: ([^\n]*)\n"
          r"USER-AGENT\s*: ([^\n]*)\n"
          r"ACCEPT\s*: ([^\n]*)\n"
          r"PROXY-CONNECTION\s*: ([^\n]*)"
}
