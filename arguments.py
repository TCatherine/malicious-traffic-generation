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
        'long': '--number',
        'short': '-n',
        'type': int,
        'choices': range(1, 1000),
        'action': 'store',
        'destination': 'number',
        'help': "Number of packets generated",
        'default': '10'
    },
    {
        'long': '--use-multiplier',
        'short': '-m',
        'type': bool,
        'action': 'store',
        'destination': 'use_multiplier',
        'help': 'The "multiplier" module will be used to generate new traffic',
        'choices': [True, False],
        'default': 'false'
    },
    {
        'long': '--use-combiner',
        'short': '-c',
        'type': bool,
        'action': 'store',
        'destination': 'use_combiner',
        'help': 'The "combiner" module will be used to generate new traffic',
        'choices': [True, False],
        'default': 'false'
    }
]
