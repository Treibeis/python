import numpy as np
import sys
import datetime

cal = [[31,28,31,30,31,30,31,31,30,31,30,31], [31,29,31,30,31,30,31,31,30,31,30,31]]
week_label = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def leap_year(yr):
	"""To see whether the input year (yr) is a leap year (1) or not (0). yr>0 means AC, yr<0 means BC, yr=0 is a wrong input."""
	if yr>0:
		if yr%4==0:
			if yr%100!=0:
				return 1
			else:
				if yr%400==0:
					return 1
				else:
					return 0
		else:
			return 0
	elif yr<0:
		if abs(yr)%4==1:
			if abs(yr)%100!=1:
				return 1
			else:
				if abs(yr)%400==1:
					return 1
				else:
					return 0
		else:
			return 0
	else:
		print('Input year error a.')
		return 0 

def date_count(yr, mo, da):
	"""To find the sequence number (count) of the input day defined by mo/da/yr (M/D/Y)."""
	if yr==0:
		print('Input year error b.')
		return -1
	else:
		t = leap_year(yr)
	if (mo<1)or(mo>12):
		print('Input month error')
		return -2
	elif (da>cal[t][mo-1])or(da<1):
		print('Input date error')
		return -3
	else:
		return np.sum([cal[t][x] for x in range(mo-1)])+da

def count_date(yr, count):
	"""To find the day of sequence number (count) in the input year (yr)."""
	t = leap_year(yr)
	if (count>sum(cal[t]))or(count<1):
		print('Input count error')
		return [0,0,0,0]
	else:
		step = 0
		edge = 31
		while edge<count:
			step += 1
			edge += cal[t][step]
		mo = step+1
		da = count-edge+cal[t][step]
		return [int(yr), int(mo), int(da), int(count)]

def main1(yr, mo, da, add):
	"""To find the day that is add day(s) after the input day defined by mo/da/yr (M/D/Y)."""
	base = date_count(yr, mo, da)
	if base<0:
		#print('Error')
		return [0,0,0,0]
	else:
		span = base+add
		if span>=0:
			step = 0
			edge = sum(cal[leap_year(yr)])
			while edge<span:
				step += 1
				edge += sum(cal[leap_year(yr+step)])
			count = span-edge+sum(cal[leap_year(yr+step)])
		else:
			step = -1
			if yr+step==0:
					step = step - 1
			edge = -sum(cal[leap_year(yr+step)])
			while edge>span:
				step -= 1
				if yr+step==0:
					step = step - 1
				edge = edge - sum(cal[leap_year(yr+step)])
			count = span-edge
		yr_ = yr+step
		out = count_date(yr_, count)
		return out
		
def main2(yr1, mo1, da1, yr2, mo2, da2):
	"""To find the time interval between the input days mo1/da1/yr1 and mo2/da2/yr2 in day."""
	t = 0 
	if yr1>yr2:
		t = 1
	elif yr1==yr2:
		if mo1>mo2:
			t = 1
		elif mo1==mo2:
			if da1>da2:
				t = 1
	if t==1:
		temp = yr1
		yr1 = yr2
		yr2 = temp
		temp = mo1
		mo1 = mo2
		mo2 = temp
		temp = da1
		da1 = da2
		da2 = temp
	c1 = date_count(yr1, mo1, da1)
	c2 = date_count(yr2, mo2, da2)
	out = 0
	for x in range(yr1, yr2):
		if x!=0:
			out += sum(cal[leap_year(x)])
	out = out+c2-c1
	return int(out)

def week(yr, mo, da):
	return int(main2(yr, mo, da, 2017, 7, 31)%7)

if __name__ == "__main__": 
	if sys.argv[1]=='h':
		print('-'*34+'Instruction'+'-'*35)
		print('Mode 0:')
		print('Input: 0 Y M D N')
		print('Output: Info of the day that is N day(s) after the day specified by M/D/Y AC.')
		print('-'*80)
		print('Mode 1:')
		print('Input: 1 Y1 M1 D1 Y2 M2 D2')
		print('Output: Time interval in day between the two days M1/D1/Y1 AC and M2/D2/Y2 AC.')
		print('-'*80)
	elif sys.argv[1]=='0':
		re = main1(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
		if re[0]>0:
			if re[3]==1:
				print('It is '+str(re[1]).zfill(2)+'/'+str(re[2]).zfill(2)+'/'+str(re[0])+' AC (MM/DD/Y), '+week_label[week(re[0],re[1],re[2])]+', which is the '+str(re[3])+'st day of',re[0],'AC.')
			elif re[3]==2:
				print('It is '+str(re[1]).zfill(2)+'/'+str(re[2]).zfill(2)+'/'+str(re[0])+' AC (MM/DD/Y), '+week_label[week(re[0],re[1],re[2])]+', which is the '+str(re[3])+'nd day of',re[0],'AC.')
			else:
				print('It is '+str(re[1]).zfill(2)+'/'+str(re[2]).zfill(2)+'/'+str(re[0])+' AC (MM/DD/Y), '+week_label[week(re[0],re[1],re[2])]+', which is the '+str(re[3])+'th day of',re[0],'AC.')
		else:
			if re[3]==1:
				print('It is '+str(re[1]).zfill(2)+'/'+str(re[2]).zfill(2)+'/'+str(abs(re[0]))+' BC, '+week_label[week(re[0],re[1],re[2])]+', which is the '+str(re[3])+'st day of',abs(re[0]),'BC.')
			elif re[3]==2:
				print('It is '+str(re[1]).zfill(2)+'/'+str(re[2]).zfill(2)+'/'+str(abs(re[0]))+' BC, '+week_label[week(re[0],re[1],re[2])]+', which is the '+str(re[3])+'nd day of',abs(re[0]),'BC.')
			else:
				print('It is '+str(re[1]).zfill(2)+'/'+str(re[2]).zfill(2)+'/'+str(abs(re[0]))+' BC, '+week_label[week(re[0],re[1],re[2])]+', which is the '+str(re[3])+'th day of',abs(re[0]),'BC.')
	elif sys.argv[1]=='1':
		re = main2(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
		print('The interval is '+str(re)+' day(s).')
	elif sys.argv[1]=='r':
		now = datetime.datetime.now()
		re1 = main2(2017, 2, 24, now.year, now.month, now.day)
		re2 = main2(1995, 10, 13, now.year, now.month, now.day)
		print('Ratio: '+str(int(re1*10000/re2)/100)+'%.')
	else:
		print('Input error.')
		exit()




