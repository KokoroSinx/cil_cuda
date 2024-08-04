include ../include.mk
OBJS = dSFMT_str_state.o dSFMT.o

mt : $(OBJS)

.SUFFIXES: .c .o 

.c.o:
	$(CC) -c $< $(CFLAGSMT) -o $@

clean:
	rm -f *~ *.o
