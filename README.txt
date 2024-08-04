INSTRUCTIONS TO COMPILE ON EBI

1. make symlink to appropriate makefile
   > ln -s makefiles/include.mk.ebi_icc include.mk

2. compile randon number generator
   > cd mt
   > make clean; make

3. compile code
   > make clean; make

4. run
   > ./cil.x input_file.json

NOTE: 
LD_LIBRARY_PATH should include the locations of the hdf5 and
(optionally) the silo libraries
Use the job.sh script inside the scripts folder 
or add the following to your .bashrc file
export LD_LIBRARY_PATH=/home/opt/visit/icc/2.10.0/linux-x86_64/lib:$LD_LIBRARY_PATH
Also, enable the latest tool set. For instance, add the following to your .bashrc file
source /opt/rh/devtoolset-4/enable -> gcc 5.2.1


DOCUMENTATION

go to directory docs and run
   > doxygen Doxyfile
