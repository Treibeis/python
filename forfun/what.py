import time
def sushu(n = 100):
	"""
	Print all the prime numbers no larger than n on the screen, n>=3
	"""
	st = time.time()
	ls = [2] # list of the prime numbers
	for a in range(3,n+1):
		t = 0
		for i in ls:
			if a%i==0:
				t = 1
		if t==0:
			ls.append(a)
	#ls = [2]+[a for a in range(3,n+1) if min([a%i for i in range(2,a)])>0]
	print('Prime numbers no larger than '+str(n)+':\n',ls)
	ed = time.time()
	print('Time taken: ',ed-st,'s')

if __name__ == "__main__":
	n = input('Input an integer larger than 1: ')
	n = int(float(n))
	if n<2:
		print('Wrong input!')
	else:
		sushu(n)
	#print('Hello, world.')
	#input("按任意键以继续...")
