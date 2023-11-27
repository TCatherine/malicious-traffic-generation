arguments = [
    {
        'long': '--type',
        'short': '-t',
        'type': str,
        'choices': ['benign', 'xss'],
        'action': 'append',
        'destination': 'types',
        'help': "Traffic type",
        'default': ['xss']
    },
    {
        'long': '--url',
        'short': '-l',
        'type': str,
        'action': 'store',
        'destination': 'url',
        'help': "Destination URL",
        'default': 'http://127.0.0.1'
    },
    {
        'long': '--number',
        'short': '-n',
        'type': int,
        'choices': range(1, 1000),
        'action': 'store',
        'destination': 'number',
        'help': "Number of packets generated",
        'default': '20'
    },
    {
        'long': '--use-multiplier',
        'short': '-m',
        'type': bool,
        'action': 'store',
        'destination': 'use_multiplier',
        'help': 'The "multiplier" module will be used to generate new traffic',
        'default': 'false'
    }
]
