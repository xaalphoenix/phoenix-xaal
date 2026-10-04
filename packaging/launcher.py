"""PyInstaller entry point (the engine process re-enters here, hence the guard)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from phoenix_stl.app import main
    sys.exit(main())
