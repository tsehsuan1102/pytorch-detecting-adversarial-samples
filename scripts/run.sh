model_name='../model/model_10.pth'
dataset='mnist'
attack='bim'
postfix=''
pic_name=$dataset'_'$attack$postfix'.png'
log_name=$dataset'_'$attack$postfix'.log'
echo 'save log to '$log_name

/usr/bin/python3 detect_adv_samples.py \
    -d $dataset \
    -a $attack \
    -m $model_name\
    --batch_size 512 \
    --do_test \
    --pic_name $pic_name | tee $log_name
