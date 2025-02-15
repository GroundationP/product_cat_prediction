import pandas as pd
import json
import pickle 

# loading files
X_Train = pd.read_csv('/data/X_train_update.csv', encoding='utf-8').rename(columns={'Unnamed: 0': 'uid'}).fillna("NULL")
Y_Train = pd.read_csv('/data/Y_train_CVw08PX.csv', encoding='utf-8').rename(columns={'Unnamed: 0': 'uid'}).fillna("NULL")
X_Train = pd.merge(X_Train, Y_Train, left_on='uid', right_on='uid', how='left')
# adding images name
main_info = X_Train[['uid', 'imageid', 'productid', 'designation', 'description', 'prdtypecode']]
main_info['image'] = main_info.apply(lambda x: 'image_' + str(x['imageid']) + '_product_' + str(x['productid']) + '.jpg', axis=1)
# generating and exporting dictionnary
key = main_info['uid'].tolist()
value = main_info[['image', 'prdtypecode', 'designation', 'description']].values.tolist()
with open('/Users/terence/A_NOTEBOOKS/Datasciencetest/PROJET_RAKUTEN/project/data/original/rak_txt_img_dict.pkl', 'wb') as f:
    pickle.dump(dict(zip(key, value)), f)

# importing dictionnary
#with open('/Users/terence/A_NOTEBOOKS/Datasciencetest/PROJET_RAKUTEN/project/data/original/rak_txt_img_dict.pkl', 'rb') as f:
#    rak_txt_img_dict = pickle.load(f)