import logging
import sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a handler if one doesn't exist for this logger
    # to avoid duplicate logs if get_logger is called multiple times.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)s)"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to the root logger if this is not the root logger itself
    # and we want to control its output independently.
    if name != "root": # Or some other specific root logger name you might use
        logger.propagate = False
        
    return logger

# Example of a root logger configuration if needed elsewhere,
# but typically applications configure this at their entry point.
# if __name__ == '__main__':
#     root_logger = get_logger("root", level=logging.DEBUG)
#     root_logger.info("Root logger configured.")
#     test_logger = get_logger("MyModule")
#     test_logger.info("This is an info message from MyModule.")
#     test_logger.debug("This is a debug message from MyModule (should not appear).")
#     test_logger.setLevel(logging.DEBUG)
#     test_logger.debug("This is a debug message from MyModule (should appear now).")
