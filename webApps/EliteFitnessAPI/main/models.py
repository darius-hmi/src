from django.db import models

class CardioExercise(models.Model):
    name = models.CharField(max_length=100)
    duration = models.PositiveIntegerField()
    calories_burnt = models.PositiveIntegerField()

    def __str__(self):
        return self.name


class WeightliftingExercise(models.Model):
    name = models.CharField(max_length=100)
    exercise_type = models.CharField(max_length=100)
    is_cross_training = models.BooleanField(default=False)
    sets = models.PositiveIntegerField()
    reps = models.PositiveIntegerField()
    weight = models.PositiveIntegerField() 

    def __str__(self):
        return self.name
