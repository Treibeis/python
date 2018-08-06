import numpy as ny

def root(a, b, c, p = 53, L = -1023, U = 1023, beta = 2.0):
	"""Optimized root-finding function for quadratic equations."""
	Max, Min = (beta-1.0/beta**(p-1))*beta**U, beta**L #Calculating the maximum and minimum of the absolute value that can be expressed with floating-point number, given p, L, U, beta.
	n = 0# Indicator of the condition of the root.
	r = []# To out put roots.
	s = 'hehe'# To describe the condition of the root.
	if a==0: #If a=0, it is possible to have one root or no root.
		if b==0:
			if c==0:
				print("Trivial.\n")
				return {}
			print("Input error.\n")
			return {}
		else:
			r.append(-c/b)
			n = 1
			return {'n':n, 'r':r, 's':'normal_1'}
	elif a<0: #For simplicity, I require that a>0.
		a, b, c = -a, -b, -c
	if b>0:	
		if c>0: #The case that c>0 is simpler.
			k = b-2.0*a**0.5*c**0.5 #To avoid meeting overflow or underflow. 
			if k<0:
				print("NO root.\n")
				return {'n':0, 'r':[], 's':'no root'}
			elif k==0: #If k-0, it is possible to have wo same roots.
				if b**0.5/a**0.5/2.0**0.5>Max**0.5:
					print("All roots overflow.\n")
					return {'n':0, 'r':[], 's':'all overflow'}
				elif b**0.5/a**0.5/2.0**0.5<Min**0.5:
					print("All roots underflow.\n")
					return {'n':0, 'r':[], 's':'all underflow'}
				else:
					r.append(b/a/2.0)
					r.append(b/a/2.0)
					return {'n':2, 'r':r, 's':'normal_2'}
			else: # k>0
				if a**0.5*c**0.5>(Max-b)/2.0: #To avoid meeting overflow.
					j = (b/6.0+1.0*a**0.5*c**0.5/3.0)
				else: #To avoid meeting underflow.
					j = (b+2.0*a**0.5*c**0.5)/6.0
				d = k**0.5*j**0.5*6.0**0.5
				if a>1: # Using the first formula for the smaller root, meanwhile guarding overflow and underflow of the root.
					if b/a/2.0<=Max-d/a/2.0:
						r.append(-b/a/2.0-d/a/2.0)
						n = n+1
				else:
					if b/2.0<=a*Max-d/2.0:
						r.append(-b/a/2.0-d/a/2.0)
						n = n+1
				if c>1: # Using the second formula for the larger root, meanwhile guarding overflow and underflow of the root.
					if b/c/2.0<=1.0/Min-d/c/2.0:
						r.append(-1.0/(b/c/2.0+d/c/2.0))	
						n = n+0.1
				else: 
					if b/2.0<=c/Min-d/2.0:
						r.append(-1.0/(b/c/2.0+d/c/2.0))	
						n = n+0.1
		elif c<0: # The method is similar with that of the case that c>0, however with more complicated operations to avoid meeting overflow or underflow.
			if (b**0.5/a**0.5/2.0**0.5)>abs(c)**0.25/a**0.25: # b^2>4ac
				if abs(c)**0.1/a**0.1<a**0.2*Min**0.1/Max**0.1/b**0.2/4.0**0.1: #If b^2/Max>4a|c|*Min, (b^2-4ac)^0.5~b, for the first formula, -b/a might be a root in the range.
					if b**0.5/a**0.5>Max**0.5:
						print("All roots overflow.\n")
						return {'n':0, 'r':[], 's':'all overflow'}
					elif b**0.5/a**0.5<Min**0.5:
						print("All roots underflow.\n")
						return {'n':0, 'r':[], 's':'all underflow'}
					else:# There must be one root underflow, since (b^2-4ac)^0.5~b.
						r.append(-b/a)
						return {'n':1, 'r':r, 's':'one root underflow'}
				else: # Using the first formula for the smaller root, meanwhile guarding overflow and underflow of the root.
					if (b**0.5/a**0.5/2.0**0.5)<=Max**0.5/(1.0+(1.0+4.0*(abs(c)**0.25*a**0.25/b**0.25/b**0.25)**4)**0.5)**0.5 and (b**0.5/a**0.5/2.0**0.5)>=Min**0.5/(1.0+(1.0+4.0*(abs(c)**0.25*a**0.25/b**0.25/b**0.25)**4)**0.5)**0.5:
						r.append(-((b**0.5/a**0.5/2.0**0.5)*(1.0+(1.0+4.0*(abs(c)**0.25*a**0.25/b**0.25/b**0.25)**4)**0.5)**0.5)**2)
						n = n+1
					elif n==0:#If the root with larger absolute value is underflow, all roots are underflow
						return {'n':0, 'r':[], 's':'all underflow'}
					if abs(a)**0.1/abs(c)**0.1<abs(c)**0.2*Min**0.1/Max**0.1/b**0.2/4.0**0.1: #If b^2/Max>4a|c|*Min, (b^2-4ac)^0.5~b,  for the second formula, -c/b might be a root in the range.
						if b**0.5/abs(c)**0.5>1.0/Min**0.5:
							print("All roots underflow.\n")
							return {'n':0, 'r':[], 's':'all underflow'}
						elif b**0.5/abs(c)**0.5<1.0/Max**0.5:
							print("All roots overflow.\n")
							return {'n':0, 'r':[], 's':'all overflow'}
						else:#There must be one root overflow.
							r.append(-c/b)
							return {'n':1, 'r':r, 's':'one root overflow'}
					else: # Using the second formula for the larger root, meanwhile guarding overflow and underflow of the root.
						if (b**0.5/abs(c)**0.5/2.0**0.5)<=(1.0/Min**0.5)/(1.0+(1.0-4.0*(abs(c)**0.25*a**0.25/b**0.25/b**0.25)**4)**0.5)**0.5 and (b**0.5/abs(c)**0.5/2.0**0.5)>=(1.0/Max**0.5)/(1.0+(1.0-4.0*(abs(c)**0.25*a**0.25/b**0.25/b**0.25)**4)**0.5)**0.5:
							r.append(((b**0.5/abs(c)**0.5/2.0**0.5)*(1.0+(1.0+4.0*(abs(c)**0.25*a**0.25/b**0.25/b**0.25)**4)**0.5)**0.5)**2)
							n = n+0.1
						elif n==0:# If the root with smaller absolute value is overflow, all roots are overflow.
							return {'n':0, 'r':[], 's':'all overflow'}
			else:# b^2<4ac
				if abs(c)**0.1/a**0.1>a**0.2*Max**0.1/Min**0.1/b**0.2/4.0**0.2:# If 4a|c|/Max>b^2/Min, (b^2-4ac)^0.5~(4a|c|)^0.5, we may have two roots as +-(|c|/a)^0.5, or no root.
					if abs(c)**0.25/a**0.25>Max**0.5:
						print("All roots overflow.\n")
						return {'n':0, 'r':[], 's':'all overflow'}
					elif abs(c)**0.25/a**0.25<Min**0.5:
						print("All roots underflow.\n")
						return {'n':0, 'r':[], 's':'all underflow'}
					else:
						r.append(-abs(c)**0.5/a**0.5)
						r.append(abs(c)**0.5/a**0.5)
						return {'n':2, 'r':r, 's':'normal_2'}
				else: # Using the first formula for the smaller root, meanwhile guarding overflow and underflow of the root.
					if (abs(c)**0.25/a**0.25)<=Max**0.5/((1.0+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**4/4.0)**0.5+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**2/2.0)**0.5 and (abs(c)**0.25/a**0.25)>=Min**0.5/((1.0+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**4/4.0)**0.5+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**2/2.0)**0.5:
						r.append(-((abs(c)**0.25/a**0.25)*((1.0+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**4/4.0)**0.5+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**2/2.0)**0.5)**2)
						n = n+1
					elif n==0:#If the root with larger absolute value is underflow, all roots are underflow
						return {'n':0, 'r':[], 's':'all underflow'}
					if abs(a)**0.1/abs(c)**0.1>b**0.2*Max**0.1/Min**0.1/abs(c)**0.2/4.0**0.1:#If 4a|c|/Max>b^2/Min, (b^2-4ac)^0.5~(4a|c|)^0.5, for the second formula, -(|c|/a)^0.5 might be a root in the range.
						if a**0.25/abs(c)**0.25>1.0/Min**0.5:
							print("All roots underflow.\n")
							return {'n':0, 'r':[], 's':'all underflow'}
						elif a**0.25/abs(c)**0.25<1.0/Max**0.5:
							print("All roots overflow.\n")
							return {'n':0, 'r':[], 's':'all overflow'}
						else: 
							#r.append(-abs(c)**0.5/a**0.5)
							r.append(abs(c)**0.5/a**0.5)
							return {'n':2, 'r':r, 's':'normal_2'}
					else: # Using the second formula for the larger root, meanwhile guarding overflow and underflow of the root.
						if (a**0.25/abs(c)**0.25)<=(1.0/Min**0.5)/((1.0+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**4/4.0)**0.5+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**2/2.0)**0.5 and (a**0.25/abs(c)**0.25)>=(1.0/Max**0.5)/((1.0+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**4/4.0)**0.5+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**2/2.0)**0.5:
							r.append(((a**0.25/abs(c)**0.25)*((1.0+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**4/4.0)**0.5+(b**0.25*b**0.25/a**0.25/abs(c)**0.25)**2/2.0)**0.5)**2)
							n = n+0.1
						elif n==0:# If the root with smaller absolute value is overflow, all roots are overflow.
							return {'n':0, 'r':[], 's':'all overflow'}
		else:# If c=0, 0 is a root, while -b/a might be a root in the range.
			if b**0.5/a**0.5<=Max**0.5:
				r.append(-b/a)
				n = n+1
			r.append(0.0)
			n = n+0.1
	elif b<0:# If b<0, we may take b\rightarrow -b, and get the sign of the obtained root (if there is any) reversed as wll.
		dd = root(a, -b, c, p, L, U, beta)
		return {'n':dd['n'], 'r':[-x for x in dd['r']], 's':dd['s']}		
	else:# b=0
		if c>0:
			print("NO root.\n")
			return {'n':0, 'r':[], 's':'no root'}	
		else:
			r.append((-c)**0.5)
			r.append((-c)**0.5)
			return {'n':2, 'r':r, 's':'normal_2'}
	if n==1:
		return {'n':1, 'r':r, 's':'one root underflow'}
	elif n==0.1:
		return {'n':1, 'r':r, 's':'one root overflow'}
	elif r==[]:
		print("Something is wrong.\n")
		return {}
	else:
		return {'n':2, 'r':r, 's':'normal_2'}
