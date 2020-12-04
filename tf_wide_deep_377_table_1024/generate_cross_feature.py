"""

Raw Feature Examples:


adplan_id:
  type: category
  transform: hash_bucket
  parameter: 10000

scheduling_id:
  type: category
  transform: hash_bucket
  parameter: 10000

Cross Feature Examples:

adplan_id&category:
  hash_bucket_size: 100
  is_deep: 1

category&ucomp:
  hash_bucket_size: 10
  is_deep: 1


"""

table_num = 377

table_size_single_FPGA = \
    [100, 100, 100, 100, 100, 100, 100, 100, 5000, 5000, 5000, 5000, 
    5000, 5000, 5000, 5000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,
    10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 
    10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 
    10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 
    30000, 30000, 30000, 30000, 100000, 100000, 100000, 100000, 100000, 100000, 
    100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 
    100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 
    100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 
    100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 
    100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 
    100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 
    100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 
    100000, 100000, 100000, 100000, 100000, 100000, 100000, 200000, 200000, 
    200000, 200000, 300000, 300000, 300000, 300000, 1000000, 1000000, 1000000, 
    1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 
    1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 
    1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 
    1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 
    1000000, 1000000, 1000000, 5000000, 5000000, 10000000, 10000000, 
    100000000, 100000000]

table_size = []
for i in range(len(table_size_single_FPGA)):
    table_size.append(table_size_single_FPGA[i])
    table_size.append(table_size_single_FPGA[i])

# the last table contain 2 * 1e8 entries
table_size.append(int(2*1e8))

assert len(table_size) == table_num
print("Table_size:", table_size)

# 14 origin features
feature = ['idea_id', 'province_id', 'city_id', 'ip_original', 'app_version', 
    'category', 'u', 'device_model', 'article_tag', 'user_cates', 'user_industrys',
    'ad_cates', 'ad_idea_types', 'ucomp']
raw_feature_num = len(feature)

# merged feature: 14 * 14 * 14
cross_feature = []
for i in range(raw_feature_num):
    for j in range(i + 1, raw_feature_num):
        for k in range(j + 1, raw_feature_num):
            if i != j and j != k:
                cross_feature.append(feature[i] + '&' + feature[j] + '&' + feature[k])

cross_feature = cross_feature[:table_num - raw_feature_num]
cross_feature_num = len(cross_feature)
print("cross_feature_num:", cross_feature_num)

table_size_raw_feature = table_size[:raw_feature_num]
table_size_cross_feature = table_size[raw_feature_num:]

### cross feature hash bucket size in yaml * 1000 = real bucket size in TF
print("\n=========== feature.yaml =============\n")
for i in range(raw_feature_num):
    print("{}:\n  type: category\n  transform: hash_bucket\n  parameter: {}\n\n".format(
        feature[i], table_size_raw_feature[i]))


print("\n=========== cross_feature.yaml ==============\n")
for i in range(cross_feature_num):
    if int(table_size_cross_feature[i]/1000) == 0:
        size = 1
    else:
        size = int(table_size_cross_feature[i]/1000)
    print("{}:\n  hash_bucket_size: {}\n  is_deep: 1\n\n".format(
        cross_feature[i], size))

# num_cross_feature = 


# ad_cates:
#   type: category
#   transform: hash_bucket
#   parameter: 1000

# ad_idea_types:
#   type: category
#   transform: hash_bucket
#   parameter: 100

# ucomp:
#   type: category
#   transform: hash_bucket
#   parameter: 1000