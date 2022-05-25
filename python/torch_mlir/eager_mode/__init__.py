import os

EAGER_MODE_DEBUG = os.environ.get("EAGER_MODE_DEBUG", 'False').lower() in ('true', '1', 't')
