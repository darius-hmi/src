from rest_framework import serializers
from .models import toDoList, Expense, Exercise, Food

class toDoListSerializer(serializers.ModelSerializer):
    class Meta:
        model = toDoList
        fields = '__all__'


class expenseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Expense
        fields = '__all__'


class exerciseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Exercise
        fields = '__all__'


class foodSerializer(serializers.ModelSerializer):
    class Meta:
        model = Food
        fields = '__all__'
