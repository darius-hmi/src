from django.shortcuts import render, redirect, get_object_or_404
from .forms import RegisterForm, PostForm, FoodForm, datePicker, ExerciseForm, toDoForm, MessageForm, ProfileForm, MealForm, MealPlanForm, ExpenseForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from .models import Post, Food, Exercise, toDoList, Message, Thread, Profile, MealPlan, Meal, Expense
from datetime import datetime, time, date as dt_date
from django.db.models import Sum, Count
from django.core.mail import send_mail
from django.utils import timezone
from django.contrib.auth.models import User
from django.http import Http404, JsonResponse
import os, requests
from django.db.models.functions import TruncDate
from django.utils.dateparse import parse_date
from django.urls import reverse
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .serializers import toDoListSerializer, expenseSerializer, exerciseSerializer, foodSerializer
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims
        token['user'] = user.username
        token['user_id'] = user.id

        return token

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer



#API code below

class toDoListViewSet(viewsets.ModelViewSet):
    queryset = toDoList.objects.all()
    serializer_class = toDoListSerializer

class expenseViewSet(viewsets.ModelViewSet):
    queryset = Expense.objects.all()
    serializer_class = expenseSerializer

class exerciseViewSet(viewsets.ModelViewSet):
    queryset = Exercise.objects.all()
    serializer_class = exerciseSerializer

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        # Check if a 'date' query parameter is provided
        date_param = request.query_params.get('date', None)
        if date_param:
            try:
                # Assuming date_param is in ISO format (e.g., 'YYYY-MM-DD')
                date = datetime.strptime(date_param, '%Y-%m-%d').date()
                queryset = queryset.filter(created_at__date=date)
            except ValueError:
                return Response({'error': 'Invalid date format. Use YYYY-MM-DD.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

class foodViewSet(viewsets.ModelViewSet):
    queryset = Food.objects.all()
    serializer_class = foodSerializer

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        # Check if a 'date' query parameter is provided
        date_param = request.query_params.get('date', None)
        if date_param:
            try:
                # Assuming date_param is in ISO format (e.g., 'YYYY-MM-DD')
                date = datetime.strptime(date_param, '%Y-%m-%d').date()
                queryset = queryset.filter(created_at__date=date)
            except ValueError:
                return Response({'error': 'Invalid date format. Use YYYY-MM-DD.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)









#Webapp backend


@login_required(login_url='/login')
def home(request):
    if request.method == 'POST':
        post_id = request.POST.get('post-id')
        if post_id:
            post = get_object_or_404(Post, id=post_id)
            # Check if the logged-in user is the author or has permission to delete the post
            if request.user == post.author:
                post.delete()
    posts = Post.objects.all()
    return render(request, 'main/home.html', {"posts": posts}) 

@login_required
def profile_view(request):
    profile, created = Profile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('profile')
    else:
        form = ProfileForm(instance=profile)
    return render(request, 'main/profile.html', {'form': form})

@login_required(login_url='/login')
def create_post(request):
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.save()
            return redirect('/home')
    else:
        form = PostForm()
    
    return render(request, 'main/create_post.html', {'form':form})


@login_required(login_url='/login')
def create_food(request):
    form = FoodForm()
    food_info = None
    error_message = None
    profile = get_object_or_404(Profile, user=request.user)
    totalCaloriesGoal = profile.total_calories_goal
    totalProteinGoal = profile.total_protein_goal

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'create_food':
            # Handle the create food form
            form = FoodForm(request.POST)
            if form.is_valid():
                food = form.save(commit=False)
                food.author = request.user
                food.save()
                return redirect('create_food')
        elif action == 'delete_food':
            # Handle food deletion
            food_id = request.POST.get('food-id')
            if food_id:
                food = Food.objects.filter(id=food_id).first()
                if food:
                    food.delete()
                    # Redirect to the same page to refresh the list
                    return redirect('create_food')
                
        elif action == 'food_lookup':
            # Handle the food lookup form
            form = FoodForm()
            food_id = request.POST.get('food_id')
            if food_id:
                api_url = f'https://api.edamam.com/api/food-database/v2/parser?ingr={food_id}&app_id={os.getenv("api_id")}&app_key={os.getenv("api_key")}'
                response = requests.get(api_url)

                if response.status_code == 200:
                    data = response.json()
                    if 'parsed' in data and data['parsed']:
                        food_info = {
                            'name': data['parsed'][0]['food']['label'],
                            'calories': data['parsed'][0]['food']['nutrients']['ENERC_KCAL'],
                            'protein': data['parsed'][0]['food']['nutrients']['PROCNT'],
                            # Add other relevant nutrient information here
                        }
                        
                        if 'hints' in data and data['hints']:
                            image_url = data['hints'][0]['food']['image']
                            food_info['image_url'] = image_url
                    else:
                        error_message = 'No food information found'

                else:
                    error_message = 'Failed to fetch food information'

            else:
                error_message = 'Please provide a food ID'
    else:
        form = FoodForm()

    author = request.user
    todaysFoods = Food.objects.filter(author=author, created_at__date=datetime.today().date())
    totalCalories = todaysFoods.aggregate(Sum('calories'))['calories__sum']
    totalProtein = todaysFoods.aggregate(Sum('protein'))['protein__sum']

    return render(request, 'main/create_food.html', {
        'form': form,
        'food_info': food_info,
        'error_message': error_message, 
        'todaysFoods': todaysFoods,
        'totalCalories': totalCalories, 
        'totalProtein': totalProtein, 
        'totalCaloriesGoal': totalCaloriesGoal,
        'totalProteinGoal': totalProteinGoal,
        })



@login_required(login_url='/login')
def foodDiary(request):
    profile = get_object_or_404(Profile, user=request.user)
    totalCaloriesGoal = profile.total_calories_goal
    totalProteinGoal = profile.total_protein_goal

    if request.method == 'POST':
        form = datePicker(request.POST)
        if form.is_valid():
            # Handle datePicker form submission
            d = form.cleaned_data['date']
            author = request.user
            todaysFoods = Food.objects.filter(author=author, created_at__date=d)
            totalCalories = todaysFoods.aggregate(Sum('calories'))['calories__sum']
            totalProtein = todaysFoods.aggregate(Sum('protein'))['protein__sum']

            return render(request, 'main/foodDiary.html', {
                'form': form,
                'todaysFoods': todaysFoods,
                'totalCalories': totalCalories,
                'totalProtein': totalProtein,
                'totalCaloriesGoal': totalCaloriesGoal,
                'totalProteinGoal': totalProteinGoal,
            })

        # Handle delete action
        form_type = request.POST.get('form-type')
        if form_type == 'delete-form':
            food_id_to_delete = request.POST.get('food-id')
            if food_id_to_delete:
                food_to_delete = Food.objects.filter(id=food_id_to_delete).first()
                food = get_object_or_404(Food, id=food_id_to_delete)
                d = food.created_at.date()
                author = request.user
                todaysFoods = Food.objects.filter(author=author, created_at__date=d)
                totalCalories = todaysFoods.aggregate(Sum('calories'))['calories__sum']
                totalProtein = todaysFoods.aggregate(Sum('protein'))['protein__sum']
                if food_to_delete:
                    food_to_delete.delete()
                    todaysFoods = Food.objects.filter(author=author, created_at__date=d)
                    totalCalories = todaysFoods.aggregate(Sum('calories'))['calories__sum']
                    totalProtein = todaysFoods.aggregate(Sum('protein'))['protein__sum']
                    # Redirect to refresh the page without the deleted entry
                    return render(request, 'main/foodDiary.html', {
                        'form': form,
                        'todaysFoods': todaysFoods,
                        'totalCalories': totalCalories,
                        'totalProtein': totalProtein,
                        'totalCaloriesGoal': totalCaloriesGoal,
                        'totalProteinGoal': totalProteinGoal,
                    })

    else:
        form = datePicker()

    return render(request, 'main/foodDiary.html', {
        'form': form,
        'totalCaloriesGoal': totalCaloriesGoal,
        'totalProteinGoal': totalProteinGoal,
    })


@login_required(login_url='/login')
def exerciseDiary(request):
    date = request.GET.get('date')
    if date:
        date = parse_date(date)
    else:
        date = dt_date.today()

    if request.method == 'POST':
        if 'form-type' in request.POST and request.POST['form-type'] == 'edit-form':
            exercise_id = request.POST.get('exercise-id')
            exercise = get_object_or_404(Exercise, id=exercise_id)
            exercise.weight = request.POST.get('weight')
            exercise.sets = request.POST.get('sets')
            exercise.reps = request.POST.get('reps')
            exercise.save()
            return redirect('exerciseDiary')

        form = datePicker(request.POST)
        if form.is_valid():
            # Handle datePicker form submission
            d = form.cleaned_data['date']
            author = request.user
            todaysExercises = Exercise.objects.filter(author=author, created_at__date=d)
            return render(request, 'main/exerciseDiary.html', {'form': form, 'todaysExercises': todaysExercises})

        # Handle delete action
        form_type = request.POST.get('form-type')
        if form_type == 'delete-form':
            exercise_id_to_delete = request.POST.get('exercise-id')
            if exercise_id_to_delete:
                exercise_to_delete = Exercise.objects.filter(id=exercise_id_to_delete).first()
                exercise = get_object_or_404(Exercise, id=exercise_id_to_delete)
                d = exercise.created_at.date()
                author = request.user
                todaysExercises = Exercise.objects.filter(author=author, created_at__date=d)
                if exercise_to_delete:
                    exercise_to_delete.delete()
                    todaysExercises = Exercise.objects.filter(author=author, created_at__date=d)
                    return render(request, 'main/exerciseDiary.html', {'form': form, 'todaysExercises': todaysExercises})
    else:
        form = datePicker(initial={'date': date})

    author = request.user
    todaysExercises = Exercise.objects.filter(author=author, created_at__date=date)
    return render(request, 'main/exerciseDiary.html', {'form': form, 'todaysExercises': todaysExercises})


@login_required(login_url='/login')
def add_Exercise(request):
    if request.method == 'POST':
        form = ExerciseForm(request.POST)
        if form.is_valid():
            exercise = form.save(commit=False)
            exercise.author = request.user
            exercise.save()
            return redirect('/addExercise')
    else:
        form = ExerciseForm()
    
    # Fetch exercise data grouped by date and muscle
    exercise_data = Exercise.objects.annotate(date=TruncDate('created_at')).values('date', 'muscle').annotate(count=Count('muscle'))
    
    return render(request, 'main/add_exercise.html', {'form':form, 'exercise_data': exercise_data})


@login_required(login_url='/login')
def get_exercise_data(request):
    exercise_data = Exercise.objects.annotate(date=TruncDate('created_at')).values('date', 'muscle').annotate(count=Count('muscle'))

    events = []
    for item in exercise_data:
        event = {
            'title': item['muscle'],
            'start': item['date'].strftime('%Y-%m-%d'),  # Convert date to string
        }
        events.append(event)

    return JsonResponse(events, safe=False)



@login_required(login_url='/login')
def add_toDoList(request):
    if request.method == 'POST':
        form = toDoForm(request.POST)
        if form.is_valid():
            todo = form.save(commit=False)
            todo.author = request.user
            todo.save()
            return redirect('/home')
    else:
        form = toDoForm()
    
    return render(request, 'main/add_toDoList.html', {'form':form})

@login_required(login_url='/login')
def viewToDoList(request):
    author = request.user
    toDoItems = toDoList.objects.filter(author=author)
    # Tested the below def so it sends an email reminder to the 
    #send_mail_test(author)
    form_type = request.POST.get('form-type')
    if form_type == 'delete-form':
        todo_id_to_delete = request.POST.get('todo-id')
        if todo_id_to_delete:
            todo_to_delete = toDoList.objects.filter(id=todo_id_to_delete).first()
            author = request.user
            toDoItems = toDoList.objects.filter(author=author)
            if todo_to_delete:
                todo_to_delete.delete()
                toDoItems = toDoList.objects.filter(author=author)
                # Redirect to refresh the page without the deleted entry
                return render(request, 'main/toDoList.html', {'toDoItems':toDoItems})


    return render(request, 'main/toDoList.html', {'toDoItems':toDoItems})


def sign_up(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('/home')
    else:
        form = RegisterForm()
    
    return render(request, 'registration/sign_up.html', {"form":form})


@login_required(login_url='/login')
def add_food(request, date):
    date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    form = FoodForm(request.POST)
    if form.is_valid():
        food = form.save(commit=False)
        food.author = request.user
        food.created_at = datetime.combine(date_obj, datetime.min.time())
        food.save()
        return redirect('foodDiary')
    else:
        form = FoodForm()

    return render(request, 'main/add_food_with_date.html', {'form': form, 'date': date})

@login_required(login_url='/login')
def add_exerciseToDate(request, date):
    date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    form = ExerciseForm(request.POST)
    if form.is_valid():
        exercise = form.save(commit=False)
        exercise.author = request.user
        exercise.created_at = datetime.combine(date_obj, datetime.min.time())
        exercise.save()
        return redirect(reverse('exerciseDiary') + '?date=' + date)
    else:
        form = ExerciseForm()

    return render(request, 'main/add_exercise_with_date.html', {'form': form, 'date': date})


#The below definition is sending an email to the yahoo account when used. Need to use Celery to send automatically
# def send_mail_test(author):
#     tomorrow = timezone.now() + timezone.timedelta(days=1)
#     tasks_due_tomorrow = toDoList.objects.filter(dueDate__range=(tomorrow, tomorrow + timezone.timedelta(days=1)), author=author)

#     for task in tasks_due_tomorrow:
#         subject =  'Test Email'
#         message = task.task + " is due tomorrow!"
#         from_email = ''
#         recipient_list = ['']

#         send_mail(subject, message, from_email, recipient_list)


@login_required
def initiate_dm(request):
    if request.method == 'POST':
        recipient_id = request.POST.get('recipient')
        recipient = get_object_or_404(User, pk=recipient_id)

        # Check if a thread between the current user and recipient already exists
        thread = Thread.objects.filter(participants=request.user).filter(participants=recipient).first()
        if thread is None:
            # If thread doesn't exist, create a new one
            thread = Thread.objects.create()
            thread.participants.add(request.user, recipient)

        return redirect('thread_detail', thread_id=thread.id)
    else:
        users = User.objects.exclude(id=request.user.id)
        return render(request, 'main/initiate_dm.html', {'users': users})



@login_required
def thread_list_view(request):
    threads = Thread.objects.filter(participants=request.user)
    return render(request, 'main/thread_list.html', {'threads': threads})

@login_required
def thread_detail_view(request, thread_id):
    try:
        thread = Thread.objects.get(pk=thread_id, participants=request.user)
    except Thread.DoesNotExist:
        raise Http404("Thread does not exist")

    messages = thread.messages.all()
    form = MessageForm()
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.save(commit=False)
            message.sender = request.user
            message.thread = thread
            message.save()
            return redirect('thread_detail', thread_id=thread.id)
    return render(request, 'main/thread_detail.html', {'thread': thread, 'messages': messages, 'form': form})



@login_required(login_url='/login')
def meal_plan_page(request):
    meal_plans = MealPlan.objects.filter(user=request.user)
    meal_plan_form = MealPlanForm()
    meal_form = MealForm()
    return render(request, 'main/meal_plan_page.html', {
        'meal_plans': meal_plans,
        'meal_plan_form': meal_plan_form,
        'meal_form': meal_form,
    })

@login_required(login_url='/login')
def create_meal_plan(request):
    if request.method == 'POST':
        form = MealPlanForm(request.POST)
        if form.is_valid():
            meal_plan = form.save(commit=False)
            meal_plan.user = request.user
            meal_plan.save()
            return JsonResponse({'status': 'success', 'meal_plan': {'id': meal_plan.id, 'name': meal_plan.name}})
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors})

@login_required(login_url='/login')
def add_meal(request, meal_plan_id):
    meal_plan = get_object_or_404(MealPlan, id=meal_plan_id, user=request.user)
    if request.method == 'POST':
        form = MealForm(request.POST)
        if form.is_valid():
            meal = form.save(commit=False)
            meal.meal_plan = meal_plan
            meal.save()
            return JsonResponse({'status': 'success', 'meal': {'day_of_week': meal.day_of_week, 'name': meal.meal_time,'contents': meal.meal_content, 'calories': meal.calories, 'protein': meal.protein}})
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors})


def delete_meal_plan(request, meal_plan_id):
    meal_plan = get_object_or_404(MealPlan, pk=meal_plan_id)
    if request.method == 'POST':
        meal_plan.delete()
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def delete_meal(request, meal_plan_id, meal_id):
    meal = get_object_or_404(Meal, pk=meal_id)
    if request.method == 'POST':
        meal.delete()
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})


@login_required(login_url='/login')
def budget_page(request):
    if request.method == 'POST':
        form = ExpenseForm(request.POST)
        if form.is_valid():
            expense = form.save(commit=False)
            expense.user = request.user
            expense.save()
            return redirect('budget_page')
    else:
        form = ExpenseForm()
    
    expenses = Expense.objects.filter(user=request.user)
    return render(request, 'main/budget.html', {'form': form, 'expenses': expenses})

#end of the code, test