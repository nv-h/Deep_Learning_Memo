
# 「scikit-learn,Keras,TensoFlowによる実践機械学習」のメモ

## docker-compose環境の構築(不要かも)

docker-composeを使用した環境が用意されているようなので、これを活用する。
現環境にはdockerはインストールされているが、docker-composeはインストールされていなかったので[Docker Compose のインストール](https://docs.docker.jp/compose/install.html)を参考にインストールする。
現時点(2021/01/13)での最新版が1.27.4だった。

```sh
sudo curl -L \
  https://github.com/docker/compose/releases/download/1.27.4/docker-compose-$(uname)-$(uname -i) \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

githubからダウンロード。ここにjupyter notebook環境が含まれる。

```sh
mkdir -p ~/sources
cd sources
git clone https://github.com/ageron/handson-ml2
cd handson-ml2
```

docker-composeでhandson-ml2環境のjupyter notebookを実行

```sh
cd docker
make build # docker-compose build
make run # docker-compose up
```

