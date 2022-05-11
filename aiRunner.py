##########################
# 라이브러리 사용
import tensorflow as tf
import pandas as pd

###########################
# 1.과거의 데이터를 준비합니다.
파일경로 = 'C:\\Users\\My PC\\Desktop\\Prime number\\data.csv'
보스턴 = pd.read_csv(파일경로)
보스턴.head(5)
# 종속변수, 독립변수
독립 = 보스턴[['x']]
종속 = 보스턴[['y']]
print(독립.shape, 종속.shape)

###########################
# 2. 모델의 구조를 만듭니다
X = tf.keras.layers.Input(shape=[1])
H = tf.keras.layers.Dense(100, activation='swish')(X)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
H = tf.keras.layers.Dense(100, activation='swish')(H)
Y = tf.keras.layers.Dense(1)(H)
checkpoint_path = "saved_model/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # 다섯 번째 에포크마다 가중치를 저장합니다
    period=1)
model.save_weights(checkpoint_path.format(epoch=0))

model = tf.keras.models.Model(X, Y)
from tensorflow.keras.losses import mean_squared_error
# 2-1
model.compile(loss="mse", optimizer='nadam', metrics='accuracy')


# 모델 구조 확인
model.summary()

###########################
# 3.데이터로 모델을 학습(FIT)합니다.
model.fit(독립, 종속, epochs=10000,,callbacks = [cp_callback])

###########################
# 4. 모델을 이용합니다
print(model.predict(독립[:5]))
print(종속[:5])

###########################
# 모델의 수식 확인
print(model.get_weights())
