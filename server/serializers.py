from rest_framework import serializers
from .models import MainObject, Interaction

class InteractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Interaction
        fields = ['request', 'response']

class MainObjectSerializer(serializers.ModelSerializer):
    interactions = InteractionSerializer(many=True, read_only=True)

    class Meta:
        model = MainObject
        fields = ['id', 'interactions']