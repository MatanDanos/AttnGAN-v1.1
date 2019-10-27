import sys
from loguru import logger
logger.remove(0)
logger.add(sys.stdout, level="INFO")
