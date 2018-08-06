import sys

def noteword(ls, fname):
	with open(fname, 'a') as f:
		for r in range(len(ls)):
			if ls[r]!="" and  ls[r]!='\n':
				f.write('['+ls[r]+']'+'['+ls[r]+']\n')
				f.write('['+ls[r]+']: https://translate.google.com/#en/zh-CN/'+ls[r].replace(' ', '%20')+'\n')


if __name__ == "__main__": 
	lw = input("Input words or expressions separated by ', ':\n")
	print (lw)
	noteword(lw.split(', '), sys.argv[1])
