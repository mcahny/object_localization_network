# Train
# bash ./tools/dist_train.sh configs/oln_box/class_agn_cascade_rpn_center.py 8 |& tee -a ./work_dirs/class_agn_crpn_ctr1_cls0/train_log.txt 
# Test
# bash ./tools/dist_test_bbox.sh configs/oln_box/class_agn_cascade_rpn_center.py work_dirs/class_agn_crpn_ctr1_cls0/epoch_8.pth 8 |& tee -a ./work_dirs/class_agn_crpn_ctr1_cls0/nonvoc.txt 

# Train -- Faster R-CNN - Ignore
bash ./tools/dist_train.sh configs/oln_box/class_agn_faster_rcnn_ignore.py 8 |& tee -a work_dirs/class_agn_faster_rcnn_ignore/train_log.txt
# Test
bash ./tools/dist_test_bbox.sh configs/oln_box/class_agn_faster_rcnn_ignore.py work_dirs/class_agn_faster_rcnn_ignore/latest.pth 8 |& tee -a work_dirs/class_agn_faster_rcnn_ignore/nonvoc_ignore.txt
