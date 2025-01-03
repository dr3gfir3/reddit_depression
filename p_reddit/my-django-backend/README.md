# My Django Backend

This project is a Django-based backend application that integrates with Apache Spark to perform data processing tasks. The application initializes a Spark session upon server startup and provides a REST API to retrieve word count information from a separate Python application.

## Project Structure

```
my-django-backend
├── my_django_backend
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── spark_app
│   ├── __init__.py
│   ├── spark_session.py
│   └── word_count.py
├── api
│   ├── __init__.py
│   ├── views.py
│   └── urls.py
├── manage.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd my-django-backend
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the server:**
   ```
   python manage.py runserver
   ```

## Usage

- The server will start and initialize a Spark session.
- You can access the API endpoints to retrieve word count information.

## API Endpoints

- **GET /api/wordcount/**: Retrieves word count information.

## Requirements

- Python 3.x
- Django
- PySpark

## License

This project is licensed under the MIT License.