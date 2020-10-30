import os, shutil

# 猫的原始数据集解压目录的路径
original_dataset_cat_dir='C:/Users/Kansx/.keras/datasets/kagglecatsanddogs/PetImages/Cat'
# 猫的原始数据集解压目录的路径
original_dataset_dog_dir='C:/Users/Kansx/.keras/datasets/kagglecatsanddogs/PetImages/Dog'

# 保存较小数据集的目录
base_dir='C:/Users/Kansx/.keras/datasets/cats_and_dogs_small'
# os.mkdir(base_dir)

train_dir=os.path.join(base_dir,'train')
# os.mkdir(train_dir)
validation_dir=os.path.join(base_dir,'validation')
# os.mkdir(validation_dir)
test_dir=os.path.join(base_dir,'test')
# os.mkdir(test_dir)

# 猫的训练图像目录
train_cats_dir=os.path.join(train_dir,'cats')
# os.mkdir((train_cats_dir))

# 狗的训练图像目录
train_dogs_dir=os.path.join(train_dir,'dogs')
# os.mkdir(train_dogs_dir)

# 猫的验证图像目录
validation_cats_dir=os.path.join(validation_dir,'cats')
# os.mkdir(validation_cats_dir)

#狗的验证图像目录
validation_dogs_dir=os.path.join(validation_dir,'dogs')
# os.mkdir(validation_dogs_dir)

# 猫的测试图像目录
test_cats_dir=os.path.join(test_dir,'cats')
# os.mkdir(test_cats_dir)

# 狗的测试图像目录
test_dogs_dir=os.path.join(test_dir,'dogs')
# os.mkdir(test_dogs_dir)

# # 将前1000张猫的图像复制到train_cats_dir
# fnames=['{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src=os.path.join(original_dataset_cat_dir,fname)
#     dst=os.path.join(train_cats_dir,fname)
#     shutil.copy(src,dst)
#
# # 将接下来500张猫的图像复制到validation_cats_dir
# fnames=['{}.jpg'.format(i) for i in range(1000,1500)]
# for fname in fnames:
#     src=os.path.join(original_dataset_cat_dir,fname)
#     dst=os.path.join(validation_cats_dir,fname)
#     shutil.copy(src,dst)
#
# # 将接下来的500张猫的图像复制到test_cats_dir
# fnames=['{}.jpg'.format(i) for i in range(1500,2000)]
# for fname in fnames:
#     src=os.path.join(original_dataset_cat_dir,fname)
#     dst=os.path.join(test_cats_dir,fname)
#     shutil.copy(src,dst)
#
# # 将前1000张狗的图像复制到train_dogs_dir
# fnames=['{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src=os.path.join(original_dataset_dog_dir,fname)
#     dst=os.path.join(train_dogs_dir,fname)
#     shutil.copy(src,dst)
#
# # 将接下来500张狗的图像复制到validation_dogs_dir
# fnames=['{}.jpg'.format(i) for i in range(1000,1500)]
# for fname in fnames:
#     src=os.path.join(original_dataset_dog_dir,fname)
#     dst=os.path.join(validation_dogs_dir,fname)
#     shutil.copy(src,dst)
#
# # 将接下来的500张狗的图像复制到test_dogs_dir
# fnames=['{}.jpg'.format(i) for i in range(1500,2000)]
# for fname in fnames:
#     src=os.path.join(original_dataset_dog_dir,fname)
#     dst=os.path.join(test_dogs_dir,fname)
#     shutil.copy(src,dst)

print('total training cat images:',len(os.listdir(train_cats_dir)))
print('total training dog images:',len(os.listdir(train_dogs_dir)))

print('total validation cat images:',len(os.listdir(validation_cats_dir)))
print('total validation dog images:',len(os.listdir(validation_dogs_dir)))

print('total test cat images:',len(os.listdir(test_cats_dir)))
print('total test dog images:',len(os.listdir(test_dogs_dir)))