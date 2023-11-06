from .reference import run as ref_run
def run(
        types: list[str],
        number: int,
        use_miltiplier: bool,
        use_combiner: bool
        ) -> list:
    reference_samples = ref_run(types, number)
    if use_miltiplier:
        # TODO: implement miltiplier module
        pass

    if use_combiner:
        # TODO: implement combiner module
        pass

    samples = reference_samples
    return samples

def main():
    # Example
    print(run(['xss'], 10, False, False))

if __name__ == "__main__":
    main()
