#!/bin/bash
for i in $( ls ../configs/Skimnames/ | grep New); do
	THEANO_FLAGS=device=gpu1 python MVAMetNN.py -config='../configs/'config$i.json
done


