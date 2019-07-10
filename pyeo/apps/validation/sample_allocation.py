from pyeo.validation import allocate_category_sample_sizes, cal_w_all, cal_sd_for_overall_accruacy, cal_sd_for_user_accuracy



se_expected_overall = 0.01 # the standard error of the estimated overall accuracy that we would like to achieve
U = {'defore': 0.7, 'gain': 0.6, 'stable_forest':0.9, 'stable_nonforest':0.95} # user's uncertainty for each class (estimated)
pixel_numbers = {'defore': 200000, 'gain':150000,'stable_forest':3200000,'stable_nonforest':6450000}
total_sample_size= 641
required_sd = 0.05 #expected user's accuracy for each class - this is larger than the sd of overall accuracy
required_val = required_sd ** 2 #variance is the root of standard error
allocate_sample = allocate_category_sample_sizes(total_sample_size=total_sample_size, user_accuracy=U, class_total_sizes= pixel_numbers,
                                                 variance_tolerance= required_val, allocate_type='olofsson')
weight = cal_w_all(pixel_numbers)
sd_overall = cal_sd_for_overall_accruacy(weight_dict=weight, u_dict = U, sample_size_dict= allocate_sample[method])
sd_u_i = cal_sd_for_user_accuracy(u_i=U[key], sample_size_i=allocate_sample[method][key])


'''
this is to calculate the total sample size for a stratified random sampling strategy, eq 13 in the Olofsson 2014 RSE paper
'''
#################
# setting variables
##################
# variables that we need to explore:

# input variable from the reference map.. will later be read from the map:

#test_n = cal_total_sample_size(se_expected_overall,U,pixels_numbers)
#test_n2 = cal_total_sample_size(se_expected_overall,U,pixels_numbers,type='full')

print('required variance for user accuracy is: ' + str(required_val))

for method in allocate_sample:
    print('---------------')
    print('stand error of overall accuracy for ' + method + ' is: ' + str(sd_overall))
    print('stand eorro of user accuracy for each class ')
    for key in allocate_sample[method]:
        print('      ' + key + ' : ' + str(sd_u_i))