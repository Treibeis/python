import sys
import argparse

def f_to_c(f):
	return (f-32.)*5./9.
	
def c_to_f(c):
	return c*9./5.+32.
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--f2c', help='input a temperature in F to get the temperature in C', type=float, nargs='?', const=100.)
	parser.add_argument('-c', '--c2f', help='input a temperature in C to get the temperature in F', type=float, nargs='?', const=40.)
	args = parser.parse_args()
	if args.f2c is not None:
		print('{:.2f} F -> {:.2f} C'.format(args.f2c, f_to_c(args.f2c)))
	if args.c2f is not None:
		print('{:.2f} C -> {:.2f} F'.format(args.c2f, c_to_f(args.c2f)))
