from django import forms
from .models import User , Attendance

class UserRegistrationForm(forms.ModelForm):
	class Meta:
		model = User
		fields = ['user_id','name', 'photo']

class Attendence_register(forms.ModelForm):
	class MAts:
		model = Attendance
		fields = ['name', 'date', 'in_time', 'out_time', 'status']
