import argparse
import logging
from arguments import arguments
import aggregator
import generator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def arguments_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generator arguments')
    for arg in arguments:
        if 'choices' in arg.keys():
            parser.add_argument(
                arg['long'],
                metavar=arg['short'],
                action=arg['action'],
                type=arg['type'],
                dest=arg['destination'],
                default=arg['default'],
                help=arg['help'],
                choices=arg['choices']
            )
        else:
            parser.add_argument(
                arg['long'],
                metavar=arg['short'],
                action=arg['action'],
                type=arg['type'],
                dest=arg['destination'],
                default=arg['default'],
                help=arg['help'],
            )
    return parser.parse_args()


def main():
    logging.info("service are started")
    args = arguments_parse()
    for _ in range(0, args.number, 10):
        samples = aggregator.run(
                    args.types,
                    10,
                    args.use_multiplier
                )
        logging.info("templates are ready")
        generator.run(
            args.url,
            samples)
        logging.info("packets are sent")

if __name__ == "__main__":
    main()
