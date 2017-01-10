#!/bin/bash
for i in $( ls ../configs/ | grep 1216); do
	THEANO_FLAGS=device=gpu1 python MVAMetNN.py -config='../configs/'$i
done
#for i in $( ls ../configs/ | grep configWith); do
	#THEANO_FLAGS=device=gpu1 python MVAMetNN.py -config='../configs/'$i
#done


