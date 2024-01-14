import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ハイパーパラメータ
batch_size = 64
learning_rate = 0.001
epochs = 10

# MNISTデータセットの読み込み
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, 
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, 
                              transform=transform, download=True)

# DataLoaderの作成
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(784, 200)
        self.l2 = nn.Linear(200, 784)

    def forward(self, x):
        # エンコーダ
        h = self.l1(x)
        # 活性化関数
        h = torch.relu(h)
        # デコーダ
        h = self.l2(h)
        # シグモイド関数で0〜1の値域に変換
        y = torch.sigmoid(h)

        return y

# モデルの設定
model = Autoencoder().to(device)

# 損失関数の設定
criterion = nn.BCELoss()

# 最適化関数の設定
optimizer = optim.Adam(model.parameters())

epochs = 10
# エポックのグループ
for epoch in range(epochs):
    train_loss = 0.
    # バッチサイズのグループ
    for (x, _) in train_dataloader:
        x = x.view(x.size(0), -1).to(device)
        # 訓練モードへの切り替え
        model.train()
        # 順伝播計算
        preds = model(x)
        # 入力画像xと復元画像predsの誤差計算
        loss = criterion(preds, x)
        # 勾配の初期化
        optimizer.zero_grad()
        # 誤差の勾配計算(バックプロバケーション)
        loss.backward()
        # パラメータの更新
        optimizer.step()
        # 訓練誤差の更新
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    print("Epochs: {}, Loss: {:.3f}".format(
         epoch+1,
         train_loss
    ))

# dataloaderからデータの取り出し
x, _ = next(iter(test_dataloader))
x = x.view(x.size(0), -1).to(device)  # バッチ内の各データを1次元に変換

# 評価モードへの切り替え
model.eval()

# 復元画像
x_rec = model(x)

# 入力画像、復元画像の表示
for i, image in enumerate([x, x_rec]):
    # 入力画像は1次元のまま表示
    if i == 0:
        image = image.view(-1, 28, 28).detach().cpu().numpy()
    # 復元画像は2次元に変更して表示
    else:
        image = image.view(-1, 28, 28).detach().cpu().numpy()
        
    plt.subplot(1, 2, i+1)
    plt.imshow(image[0], cmap="binary_r")  # 2次元の場合はimage[0]で最初の画像を取得
    plt.axis("off")
plt.show()
