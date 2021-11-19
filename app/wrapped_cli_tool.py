import os

import click
import streamlit.cli


@click.group()
def main():
    pass


@main.command("streamlit")
def main_streamlit():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "main.py")
    args = []
    streamlit.cli._main_run(filename, args)


if __name__ == "__main__":
    main()
