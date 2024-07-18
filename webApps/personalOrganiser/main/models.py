from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver
# Create your models here.


class Food(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    protein = models.IntegerField()
    calories = models.IntegerField()
    created_at = models.DateTimeField(default=timezone.now)
    

    def __str__(self):
        return self.name



class Post(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title + '\n' + self.description

class Exercise(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    class names(models.TextChoices):
        flatBenchPressBB = 'Flat barbell bench press', 'Flat barbell bench press',
        inclineBenchPressBB = 'Incline barbell bench press', 'Incline barbell bench press',
        flatBenchPressDB = 'flat dumbbell Bench Press','flat dumbbell Bench Press',
        inclineBenchLowAngleDB = 'Incline dumbbell press low angle','Incline dumbbell press low angle',
        inclineBenchhighAngleDB = 'Incline dumbbell press high angle','Incline dumbbell press high angle',
        flyesDB = 'Dumbbell flyes', 'Dumbbell flyes',
        FlyesCable = 'Cable flyes', 'Cable flyes',
        machineFlyes = 'Machine flyes','Machine flyes',
        machineChestPress = 'Machine Chest Press','Machine Chest Press',

        shoulderPressDB = 'Dumbbell shoulder press', 'Dumbbell shoulder press',
        shoulderPressBB = 'Barbell Dumbbell press','Barbell Dumbbell press',
        lateralRaisesCable = 'Cable lateral raises','Cable lateral raises',
        lateralRaisesDB = 'Dumbbell lateral raises','Dumbbell lateral raises',
        frontRaisesDB = 'Dumbbell Front Raises','Dumbbell Front Raises',
        frontRaisesCable = 'Cable Front Raises','Cable Front Raises',
        frontRaises = 'Front Raises','Front Raises',
        rearDeltMachine = 'Rear Delt Machine','Rear Delt Machine',
        rearDeltDB = 'Rear Delt Dumbbell','Rear Delt Dumbbell',
        machineShoulderPress = 'Machine shoulder press','Machine shoulder press',
        trapsRaises = 'Traps raises','Traps raises',
    
        bentOverRowBB = 'Barbell Bent over row','Barbell Bent over row',
        ropePulldown = 'Rope Pulldown','Rope Pulldown',
        deadlift = 'Deadlift', 'Deadlift',
        latPulldownStraightBar = 'Lat pulldown straight bar','Lat pulldown straight bar',
        latPulldownCloseGrip = 'Lat pulldown close grip','Lat pulldown close grip',
        cableRowWide = 'Cable row wide', 'Cable row wide', 
        cableRowNarrow = 'Cable row narrow', 'Cable row narrow',
        singleArmLatPulldown = 'Single arm lat pulldown','Single arm lat pulldown',
        machineRow = 'Machine Row', 'Machine Row',
        pullUps = 'pull ups','pull ups',
    
        squats = 'Squats','Squats',
        legPress = 'Leg press', 'Leg press',
        LegExtension = 'Leg extension', 'Leg extension',
        LegCurls = 'Leg curls','Leg curls',
        bulgarianSquats = 'Bulgarian split squats','Bulgarian split squats',
        rdl = 'Romian deadlift','Romian deadlift',

        bicepCurls = 'Bicep curls','Bicep curls',
        hammerCurls = 'Hamemr curls','Hamemr curls',
        bicepCurlCable = 'Cable bicep curls', 'Cable bicep curls', 
        hammerCurlsCable = 'Cable hammer curl','Cable hammer curl',
        underhandLatPulldown = 'underhand lat pulldown','underhand lat pulldown',

        tricepExtentionCable = 'Cable tricep extension','Cable tricep extension',
        tricepExtensionBar = 'Bar tricep extension','Bar tricep extension',
        overheadExtnsion = 'Overhead extension','Overhead extension',
        dips = 'Dips','Dips',



    name = models.CharField(
        max_length=50,
        choices = names.choices
    )

    weight = models.IntegerField()
    sets = models.IntegerField()
    reps = models.IntegerField()
    class muscleGroup(models.TextChoices):
        Chest = 'Chest', 'Chest'
        Shoulder = 'Shoulder', 'Shoulder'
        Back = 'Back', 'Back'
        Legs = 'Legs', 'Legs'
        Arms = 'Arms', 'Arms'
    muscle = models.CharField(
        max_length=10,
        choices = muscleGroup.choices
    )
    created_at = models.DateTimeField(default=timezone.now)
    def __str__(self):
        return self.name


class toDoList(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    dueDate = models.DateField()
    task = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.task

class Thread(models.Model):
    participants = models.ManyToManyField(User, related_name='threads')


class Message(models.Model):
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name='messages')
    sender = models.ForeignKey(User, related_name='sent_messages', on_delete=models.CASCADE)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    weight = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    total_calories_goal = models.IntegerField(null=True, blank=True)
    total_protein_goal = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} Profile'

class MealPlan(models.Model):
    name = models.CharField(max_length=100)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class Meal(models.Model):
    DAY_CHOICES = [
        ('Monday', 'Monday'),
        ('Tuesday', 'Tuesday'),
        ('Wednesday', 'Wednesday'),
        ('Thursday', 'Thursday'),
        ('Friday', 'Friday'),
        ('Saturday', 'Saturday'),
        ('Sunday', 'Sunday'),
    ]
    day_of_week = models.CharField(max_length=10, choices=DAY_CHOICES)
    meal_time = models.CharField(max_length=100)
    meal_content = models.CharField(max_length=100)
    meal_plan = models.ForeignKey(MealPlan, on_delete=models.CASCADE, related_name='meals')
    calories = models.DecimalField(max_digits=5, decimal_places=2)
    protein = models.DecimalField(max_digits=5, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} on {self.day_of_week}"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()


class Expense(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.PositiveIntegerField()

    def __str__(self):
        return f'{self.name} - {self.amount}'

