'''
this is to calculate the total sample size for a stratified random sampling strategy, eq 13 in the Olofsson 2014 RSE paper
'''
import numpy as np
import pdb
#################
# setting variables
##################
# variables that we need to explore:
se_expected_overall = 0.01 # the standard error of the estimated overall accuracy that we would like to achieve
U = {'defore': 0.7, 'gain': 0.6, 'stable_forest':0.9, 'stable_nonforest':0.95} # user's uncertainty for each class (estimated)

# input variable from the reference map.. will later be read from the map:
pixel_numbers = {'defore': 200000, 'gain':150000,'stable_forest':3200000,'stable_nonforest':6450000}


def cal_si(ui):
    si = np.sqrt(ui * (1 - ui))
    return si


def cal_wi(n,total_n):
    wi = float(n/total_n)
    return wi


def cal_w_all(dict_pixel_numbers):
    w_dict = {}
    total_pixel = (sum(dict_pixel_numbers.values()))
    for key in dict_pixel_numbers:
        w_dict[key] = cal_wi(n=dict_pixel_numbers[key], total_n= total_pixel)
    return w_dict


def cal_n_by_prop(weight, sample_size):
    n = round(weight * sample_size)
    return n


def val_to_sd(val):
    sd = val**0.5
    return sd


def cal_val_for_overall_accruacy(weight_dict,u_dict,sample_size_dict):
    sum_val = 0
    for key in u_dict:
        val_i = (weight_dict[key] **2) * u_dict[key] * (1-u_dict[key])/(sample_size_dict[key]-1)
        sum_val += val_i
    return sum_val


def cal_val_for_user_accuracy(u_i,sample_size_i):
    val_user = (u_i*(1-u_i))/(sample_size_i-1)
    return val_user


def cal_sd_for_overall_accruacy(weight_dict,u_dict,sample_size_dict):
    val_overall = cal_val_for_overall_accruacy(weight_dict=weight_dict,u_dict=u_dict,sample_size_dict=sample_size_dict)
    sd_overall = val_to_sd(val_overall)
    return sd_overall


def cal_sd_for_user_accuracy(u_i,sample_size_i):
    val_user = cal_val_for_user_accuracy(u_i=u_i,sample_size_i=sample_size_i)
    sd_user = val_to_sd(val_user)
    return sd_user

def cal_total_sample_size(se_expected_overall, U, pixel_numbers, type = 'simple'):
    total_pixel = (sum(pixel_numbers.values()))
    if type == 'simple':
        weighted_U_sum = 0
        # weight are equal between different classes
        for key in U:
            S_i = cal_si(U[key])
            Wi = cal_wi(n= pixel_numbers[key], total_n=total_pixel)# proportion of each class
            weighted_U_sum += S_i*Wi
        n = (weighted_U_sum/se_expected_overall)**2
    elif type == 'full':
        weighted_U_sum2 = 0
        weighted_U_sum = 0
        # weight are equal between different classes
        for key in U:
            S_i = cal_si(U[key])
            Wi = cal_wi(n= pixel_numbers[key], total_n=total_pixel)  # proportion of each class
            weighted_U_sum2 += S_i * Wi
            weighted_U_sum += (S_i ** 2) * Wi
        up = (weighted_U_sum2) ** 2
        bottom_right = (1 / total_pixel) * weighted_U_sum

        n = (up / (se_expected_overall ** 2 + bottom_right))
    print('suggested total sample size are:' + str(n))
    return n


#test_n = cal_total_sample_size(se_expected_overall,U,pixels_numbers)
#test_n2 = cal_total_sample_size(se_expected_overall,U,pixels_numbers,type='full')

total_sample_size= 641
required_sd = 0.05 #expected user's accuracy for each class - this is larger than the sd of overall accuracy
required_val = required_sd ** 2 #variance is the root of standard error
print('required variance for user accuracy is: ' + str(required_val))


def cal_minum_n(expected_accuracy,required_val):
    n = expected_accuracy*(1-expected_accuracy)/required_val
    return n


def allocate(total_sample_size, user_accuracy,pixel_numbers, required_val, allocate_type= 'olofsson'):
    minum_n = {}
    allocated_n = {}
    weight = cal_w_all(pixel_numbers)
    print('the weight for each class is: ')
    print(weight)
    print('-----------------')
    print('the minimum sampling number for : ')
    for key in user_accuracy:
        minum_n_i = cal_minum_n(expected_accuracy=U[key], required_val=required_val)
        print('      ' + key + ' is: ' + str(round(minum_n_i)))
        minum_n[key] = minum_n_i

    if allocate_type == 'equal' or allocate_type == 'prop':
        for key in user_accuracy:
            if allocate_type == 'equal':
                n = total_sample_size/float(len(user_accuracy.keys()))
            elif allocate_type == 'prop':
                n = cal_n_by_prop(weight =weight[key], sample_size=total_sample_size)
            else:
                continue
            allocated_n[key] = n
            print('allocated sampling number for ' + key + ' is: ' + str(allocated_n[key]))

    if allocate_type == 'olofsson':
        pre_allocated_n = {'alloc1': {'defore': 100, 'gain': 100},
                           'alloc2': {'defore': 75, 'gain': 75},
                           'alloc3': {'defore': 50, 'gain': 50}}

        for method in pre_allocated_n:  # for each allocation method
            already_allocated = sum(pre_allocated_n[method].values())
            remaining_sample_size = total_sample_size - already_allocated

            w_stable_forest = weight['stable_forest']/(weight['stable_forest'] + weight['stable_nonforest'] )

            w_stable_nonforest = weight['stable_nonforest']/(weight['stable_nonforest'] + weight['stable_forest'])
            pre_allocated_n[method]['stable_forest'] = cal_n_by_prop(w_stable_forest, remaining_sample_size)
            pre_allocated_n[method]['stable_nonforest'] = cal_n_by_prop(w_stable_nonforest, remaining_sample_size)

            allocated_n[method] = pre_allocated_n[method]
        print('allocated sample number under different scenario is: ')
        print(allocated_n)
    return allocated_n


allocate_sample = allocate(total_sample_size=total_sample_size,user_accuracy=U, pixel_numbers = pixel_numbers,required_val = required_val, allocate_type= 'olofsson')

weight = cal_w_all(pixel_numbers)

for method in allocate_sample:
    sd_overall = cal_sd_for_overall_accruacy(weight_dict=weight, u_dict = U, sample_size_dict= allocate_sample[method])
    print('---------------')
    print('stand error of overall accuracy for ' + method + ' is: ' + str(sd_overall))
    print('stand eorro of user accuracy for each class ')
    for key in allocate_sample[method]:
        sd_u_i = cal_sd_for_user_accuracy(u_i=U[key], sample_size_i=allocate_sample[method][key])
        print('      ' + key + ' : ' + str(sd_u_i))


