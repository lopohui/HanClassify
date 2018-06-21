data_dir=../data
data=$data_dir/yelp_academic_dataset_review.json
data_random=$data_dir/data.random
data_train=$data_dir/train
data_dev=$data_dir/dev
data_test=$data_dir/test
shuf $data > $data_random
data_num=`wc -l $data_random`
train_num=$(awk -v num="$data_num" 'BEGIN{print int(num*0.8+0.5)}' )
dev_num=$(awk -v num="$data_num" "BEGIN{print int(num*0.1+0.5)}")

split -l $train_num $data_random -d -a 1 data_
mv data_0 $data_train

split -l $dev_num data_1 -d -a 2 data_
rm data_1
mv data_00 $data_dev
mv data_01 $data_test



