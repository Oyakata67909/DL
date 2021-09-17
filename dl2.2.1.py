#CIFAR10のデータをCNNでクラス分類

#datasetの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
y_train = np.eye(10)[y_train.astype('int32').flatten()]

x_test = x_test.astype('float32') / 255
y_test = np.eye(10)[y_test.astype('int32').flatten()]

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=10000)

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05,
                    wspace=0.05)

for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i])

#実装
model = Sequential()

model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal', input_size=(32, 32, 3)))
#32x32x3 -> 28x28x6
model.add(MaxPooling2D(pool_size=(2, 2))) #28x28x6 -> 14x14x6
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal'))
# 14x14x6 -> 10x10x16

model.add(MaxPooling2D(pool_size=(2, 2)))
#10x10x16 -> 5x5x16

model.add(Flatten()) #5x5x16 -> 400
model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
#400 => 120
model.add(Dense(84, activation='relu', kernel_initializer='he_normal'))
#120 -> 84
model.add(Dense(10, activation='softmax')) # 84 -> 10

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metircs=['accuracy'])

early_stopping = EarlyStopping(patience=1, verbose=1)
model.fit(x=x_trian, y=y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_valid, y_valid), callbacks=[early_stopping])
