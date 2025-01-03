from django.core.asgi import get_asgi_application
import os
from spark_app.spark_session import create_spark_session

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_django_backend.settings')

# Create Spark session when the server starts
spark = create_spark_session()

application = get_asgi_application()