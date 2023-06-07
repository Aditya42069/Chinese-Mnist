import numpy as np
import tensorflow_datasets as tfds
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from skimage import io, transform
df = pd.read_csv("archive/chinese_mnist.csv")
df['filename'] = ("archive/data/data/""input_" + df['suite_id'].astype(str)+"_" +df['sample_id'].astype(str)+"_"+df['code'].astype(str)+ ".jpg")
d_test=df.sample(n=1000,random_state=1)
for i in list(d_test.index):
    df.drop(i,axis=0,inplace=True)
d_train=df
def generate_datasets(df):
    images = []
    for filename in tqdm(df['filename']):
        image = io.imread(filename)
        image = transform.resize(image, (64,64,1))
        images.append(image)
    images = np.array(images)
    df = pd.get_dummies(df['character'])
    return images, df
d_train=d_train.sample(frac=1)
d_test=d_test.sample(frac=1)
train_img,train_df_temp=generate_datasets(d_train)
test_img,test_df_temp=generate_datasets(d_test)
train_df=np.sum(train_df_temp*range(0,15),axis=1)
test_df=np.sum(test_df_temp*range(0,15),axis=1)
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64)),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_img, train_df,batch_size=32, epochs=50)
model.evaluate(test_img,test_df)
