import requests
import random
import logging
import time
from .url_params import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def send_request(url: str, endpoint: str):
    endpoint = url + endpoint
    method = 'GET'
    headers = {
        'User-Agent': random.choice(user_agent_list),
        'Accept-Encoding': random.choice(accept_encoding_list),
        'Accept': random.choice(accept_list),
        'Connection': random.choice(connection_list),
    }
    try:
        if method == 'GET':
            logging.info(f"send: {endpoint}")
            requests.get(url=endpoint, headers=headers)
        else:
            logging.info(f"send: {endpoint}")
            requests.post(url=endpoint, headers=headers)
    except:
        logging.debug(f"An exception occurred")


def run(url: str, samples: list):
    for sample in samples:
        send_request(url, sample)
        time.sleep(1)


def main():
    # Example
    samples = [
        (
            'GET',
            '/search.php?test=query%3C%3CSCRIPT%3Ealert(%22XSS%22);//%3C%3C/SCRIPT%3E%0A ',
            '1.1',
            'test',
            'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
            'gzip, deflate',
            '*/*',
            'keep-alive'
        )
    ]
    print(run(samples))


if __name__ == "__main__":
    main()
