from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Post, Food, Exercise, toDoList, Message, Profile, Meal, MealPlan, Expense

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'description']


class FoodForm(forms.ModelForm):
    class Meta:
        model = Food
        fields = ['name', 'calories', 'protein']

class toDoForm(forms.ModelForm):
    class Meta:
        model = toDoList
        fields = ['task', 'dueDate']
        labels = {'dueDate': 'Due Date'}
        widgets = {'dueDate':forms.DateInput(
            attrs={'type': 'date'}
        )}

class DateInput(forms.DateInput):
    input_type = 'date'

class datePicker(forms.Form):
    date = forms.DateField(widget=DateInput)


class ExerciseForm(forms.ModelForm):
    class Meta:
        model = Exercise
        fields = ['muscle','name', 'weight', 'sets', 'reps']


class MessageForm(forms.ModelForm):
    class Meta:
        model = Message
        fields = ['content']

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['weight', 'height', 'total_protein_goal', 'total_calories_goal']

class MealPlanForm(forms.ModelForm):
    class Meta:
        model = MealPlan
        fields = ['name']

class MealForm(forms.ModelForm):
    class Meta:
        model = Meal
        fields = ['day_of_week', 'meal_time', 'meal_content', 'calories', 'protein']
        labels = {'meal_time': 'Meal time/number'}
    

class ExpenseForm(forms.ModelForm):
    class Meta:
        model = Expense
        fields = ['name', 'amount', 'date']
        widgets = {
            'date': forms.NumberInput(attrs={'min': 1, 'max': 31}),
        }
        labels = {'date': 'day of the month'}