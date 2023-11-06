from .locals import xss, benign
from .parser import parse

def run(
        types: list[str],
        number: int,
):
    data = {
        'xss': xss,
        'benign': benign
    }
    type_number = number // len(types)

    samples = []
    for type in types:
        samples.extend(parse(data[type], type_number))
    return samples


def main():
    # Example
    print(run(['xss'], 10))

if __name__ == "__main__":
    main()
