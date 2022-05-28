from loguru import logger

logger.add(
    "{time:dddd MMMM YYYY}.log",
    level="DEBUG",
    format="{time:YYYY:MMMM:dddd:H:mm:ss} | {level} | {message}",
    rotation="1 days",
)