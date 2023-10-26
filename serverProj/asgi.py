"""
ASGI config for myproject project.

It exposes the ASGI callable as a module-level variable named ``application``.
"""

import os
from django.core.asgi import get_asgi_application

# Fetch the Django ASGI application early to ensure the AppRegistry is populated before importing other components.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django_asgi_application = get_asgi_application()

# Import other necessary ASGI components here (e.g., channels)

# Define the ASGI application (standard for Django, modify as needed for channels, etc.)
application = django_asgi_application