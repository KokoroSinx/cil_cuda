#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -o std.out -e std.err
#$ -q ara.q
#$ -pe openmp 8   #並列環境と使用プロセス数の指定
#$ -v OMP_NUM_THREADS=8 # スレッド数の指定（環境変数を指定する場合は -v　を使う）

#$ -v LD_LIBRARY_PATH=//opt/hdf5/1.12.0/lib

# run
echo $JOB_ID "@" $HOSTNAME > $JOB_NAME.log
/home/jintao/cilebi/CIL/cil.x test.2d.json