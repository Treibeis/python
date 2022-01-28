.PHONY : clean

libcheminet.so : cheminet.o
	gcc -shared -Wl,-soname,libcheminet.so -o libcheminet.so cheminet.o

cheminet.o : cheminet.c
	gcc -c -fPIC cheminet.c -o cheminet.o

clean :
	-rm -vf libcheminet.so cheminet.o cheminet.pyc
