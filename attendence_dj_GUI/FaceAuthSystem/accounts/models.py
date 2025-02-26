from django.db import models
from django.utils.timezone import now

class User(models.Model):
	user_id = models.CharField(max_length=100,default='')
	name = models.CharField(max_length=100)
	date = models.DateField(default=now)
	photo = models.ImageField(upload_to='')

	def __str__(self):
		return self.name

class Attendance(models.Model):
	user_id = models.CharField(max_length=100,default='')
	name = models.CharField(max_length=100)
	date = models.DateField()
	in_time = models.TimeField(null=True, blank=True)
	out_time = models.TimeField(null=True, blank=True)
	# status = models.CharField(max_length=20, default='Present')
	class meta:
		unique_together = ('date')
	def __str__(self):
		return f"{self.name} - {self.date}"