import argparse

from arguments import arguments
import aggregator
import generator

def arguments_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generator arguments')
    for arg in arguments:
        parser.add_argument(
            arg['long'],
            metavar=arg['short'],
            action=arg['action'],
            type=arg['type'],
            dest=arg['destination'],
            default=arg['default'],
            help=arg['type'],
            choices=arg['choices']
        )
    return parser.parse_args()


def main():
    args = arguments_parse()
    samples = aggregator.run(
                args.types,
                args.number,
                args.use_multiplier,
                args.use_combiner
            )
    generator.run(samples)

if __name__ == "__main__":
    main()
