
 for d in 0.2
 do
 	PD=graph_data/douban
 	TRAINRATIO=${d}
 	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
   TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict

 	python -u network_alignment.py \
     --source_dataset ${PD}/online/graphsage/ \
     --target_dataset ${PD}/offline/graphsage/ \
     --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
 	CINA \
 	--train_dict ${TRAIN} \
 	--cuda
 done

#for d in 0.2
#do
#	PD=graph_data/acm_dblp
#	TRAINRATIO=${d}
#	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
#    TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict
#
#	python -u network_alignment.py \
#    --source_dataset ${PD}/acm/graphsage/ \
#    --target_dataset ${PD}/dblp/graphsage/ \
#    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
#	CINA \
#	--train_dict ${TRAIN} \
#	--lr 0.00008\
#	--cuda
#done

#for d in 0.2
#do
#	PD=graph_data/fb-tw-data
#	TRAINRATIO=${d}
#	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
#    TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict
#
#	python -u network_alignment.py \
#    --source_dataset ${PD}/facebook/graphsage/ \
#    --target_dataset ${PD}/twitter/graphsage/ \
#    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
#	CINA \
#	--train_dict ${TRAIN} \
#	--cuda
#done

#for d in 0.2
#do
#	PD=graph_data/flickr_myspace
#	TRAINRATIO=${d}
#	TRAIN=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
#    TEST=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict
#
#	python -u network_alignment.py \
#    --source_dataset ${PD}/flickr/graphsage/ \
#    --target_dataset ${PD}/myspace/graphsage/ \
#    --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
#	CINA \
#	--train_dict ${TRAIN} \
#	--lr 0.015\
#	--cuda
#done

