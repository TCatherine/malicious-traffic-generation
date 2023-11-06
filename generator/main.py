import requests

URL = 'http://127.0.0.1'


def send_request(parameters):
    endpoint = URL + parameters[1]
    method = parameters[0]
    headers = {
        'User-Agent': parameters[4],
        'Accept-Encoding': parameters[5],
        'Accept': parameters[6],
        'Connection': parameters[7]
    }
    try:
        if method == 'GET':
            requests.get(url=endpoint, headers=headers)
        else:
            requests.post(url=endpoint, headers=headers)
    except:
        print("An exception occurred")


def run(samples: list):
    for sample in samples:
        send_request(sample)


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
