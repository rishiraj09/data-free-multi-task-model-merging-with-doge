"""Write an environment record (ENV-03) to results/manifests and print its path."""

from doge_exam.env_info import write_environment_record
from doge_exam.paths import load_paths


def main() -> None:
    paths = load_paths()
    paths.ensure()
    print(write_environment_record(paths.manifests))


if __name__ == "__main__":
    main()
