#!/bin/bash
# 手动下载BERT模型到本地

echo "开始下载 BERT-base-uncased 模型..."

# 创建缓存目录
CACHE_DIR=~/.cache/huggingface/hub
mkdir -p $CACHE_DIR

# 使用国内镜像下载
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型文件
cd $CACHE_DIR
wget -c https://hf-mirror.com/bert-base-uncased/resolve/main/pytorch_model.bin
wget -c https://hf-mirror.com/bert-base-uncased/resolve/main/config.json
wget -c https://hf-mirror.com/bert-base-uncased/resolve/main/vocab.txt
wget -c https://hf-mirror.com/bert-base-uncased/resolve/main/tokenizer_config.json

echo "下载完成！"
echo "模型位置: $CACHE_DIR"

