# # Train 22 -- rpncls
# bash ./tools/dist_train.sh configs/oln_box/oln_box_rpncls_frcls_22.py 8

# Train 21 -- center_rpncls
bash ./tools/dist_train.sh configs/oln_box/oln_box_center_biou_rpncls_fr_cls_21.py 8
# Train 20 -- center_rpncls
bash ./tools/dist_train.sh configs/oln_box/oln_box_center_biou_rpncls_20.py 8

# # Train 19 -- center
# bash ./tools/dist_train.sh configs/oln_box/oln_box_center_biou_frcls_20.py 8
# # Train 18 -- center
# bash ./tools/dist_train.sh configs/oln_box/oln_box_biou_center_18.py 8
# # Re-run done -- Train 17 -- biou
# bash ./tools/dist_train.sh configs/oln_box/oln_box_center_center_17.py 8
# # Train 16 -- biou
# bash ./tools/dist_train.sh configs/oln_box/oln_box_biou_biou_16.py 8
# # Train 15 -- center
# bash ./tools/dist_train.sh configs/oln_box/oln_box_center_biou_15.py 8