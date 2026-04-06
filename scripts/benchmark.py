"""Run the local late-interaction benchmark after installing the package."""


def main() -> None:
    try:
        from late_interaction.bench import main as benchmark_main
    except ImportError as error:
        raise SystemExit("Install the package or run with PYTHONPATH=src.") from error

    benchmark_main()


if __name__ == "__main__":
    main()
