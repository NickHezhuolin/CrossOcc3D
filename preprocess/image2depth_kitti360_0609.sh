set -e
exeFunc(){
    cd /home/hez4sgh/1_work_dir/CGFormer/preprocess/mobilestereonet
    baseline=$1
    num_seq=$2
    data_path=$3
    CUDA_VISIBLE_DEVICES=2 python prediction.py --datapath $data_path/data_2d_raw/$num_seq \
    --testlist /home/hez4sgh/1_work_dir/CGFormer/preprocess/mobilestereonet/filenames_kitti360/$num_seq.txt --num_seq $num_seq --loadckpt /home/hez4sgh/1_work_dir/CGFormer/preprocess/mobilestereonet/MSNet3D_SF_DS_KITTI2015.ckpt --dataset kitti360 \
    --model MSNet3D --savepath $data_path/depth --baseline $baseline
    cd ..
}

data_path='/home/hez4sgh/1_work_dir/CGFormer/data/SSCBenchKITTI360'

for num_seq in '2013_05_28_drive_0006_sync' '2013_05_28_drive_0009_sync'
do
    exeFunc 331.5326 $num_seq $data_path
done